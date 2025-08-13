"""
Base functionality for torch-projectors benchmarking.

This module provides common utilities for all benchmark scripts including:
- Platform detection and system information gathering
- JSON I/O for results storage
- Statistical analysis helpers
- Timing and memory measurement utilities
- Common test data generation
"""

import json
import time
import platform
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse

import torch
import psutil


class BenchmarkBase:
    """Base class for all benchmarking scripts."""
    
    def __init__(self, platform_name: str, experiment_name: str):
        self.platform_name = platform_name
        self.experiment_name = experiment_name
        self.device = self._detect_device()
        self.results_dir = Path(__file__).parent.parent / "data" / experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.results_dir / f"{platform_name}.json"
        
        # Benchmark configuration
        self.warmup_runs = 5
        self.timing_runs = 20
        self.outlier_threshold = 2.0  # Standard deviations
        
        # Initialize results structure
        self.results = {
            "metadata": self._gather_system_info(),
            "benchmarks": {}
        }
    
    def _detect_device(self) -> torch.device:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    @staticmethod
    def parse_device(device_str: str) -> torch.device:
        """Parse device string and return appropriate torch.device."""
        if device_str == 'auto':
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        elif device_str == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            return torch.device("cuda")
        elif device_str == 'mps':
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                raise RuntimeError("MPS requested but not available")
            return torch.device("mps")
        elif device_str == 'cpu':
            return torch.device("cpu")
        else:
            raise ValueError(f"Unknown device: {device_str}")
    
    def _gather_system_info(self) -> Dict[str, Any]:
        """Collect comprehensive system and environment information."""
        import torch_projectors
        
        metadata = {
            "platform_name": self.platform_name,
            "timestamp": datetime.now().isoformat(),
            "torch_projectors_version": getattr(torch_projectors, '__version__', 'unknown'),
            "torch_version": torch.__version__,
            "device_info": self._get_device_info(),
            "system_info": self._get_system_info()
        }
        return metadata
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information."""
        device_info: Dict[str, Any] = {
            "type": self.device.type
        }
        
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            device_info["name"] = torch.cuda.get_device_name(self.device)
            device_info["memory_gb"] = float(props.total_memory / 1e9)
            device_info["compute_capability"] = f"{props.major}.{props.minor}"
        elif self.device.type == "mps":
            device_info["name"] = "Apple Metal Performance Shaders"
            device_info["memory_gb"] = float(psutil.virtual_memory().total / 1e9)  # Unified memory
        else:  # CPU
            device_info["name"] = platform.processor() or "Unknown CPU"
            device_info["cores"] = psutil.cpu_count(logical=False) or 1
            device_info["threads"] = psutil.cpu_count(logical=True) or 1
            device_info["memory_gb"] = float(psutil.virtual_memory().total / 1e9)
        
        return device_info
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        memory = psutil.virtual_memory()
        return {
            "platform": platform.platform(),
            "cpu": platform.processor() or "Unknown",
            "memory_gb": float(memory.total / 1e9),
            "python_version": platform.python_version(),
            "architecture": platform.machine()
        }
    
    def generate_test_data(self, batch_size: int, height: int, width: int, 
                          num_projections: int, is_3d: bool = False, 
                          depth: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate consistent test data for benchmarks."""
        torch.manual_seed(42)  # Fixed seed for reproducibility
        
        if is_3d:
            if depth is None:
                depth = height  # Default to cubic volumes
            # 3D volumes in RFFT format: [batch, depth, height, width//2 + 1]
            reconstructions = torch.randn(
                batch_size, depth, height, width // 2 + 1, 
                dtype=torch.complex64, device=self.device
            )
            # 3D rotation matrices [batch, num_projections, 3, 3]
            rotations = self._generate_3d_rotations(batch_size, num_projections)
        else:
            # 2D reconstructions in RFFT format: [batch, height, width//2 + 1]  
            reconstructions = torch.randn(
                batch_size, height, width // 2 + 1,
                dtype=torch.complex64, device=self.device
            )
            # 2D rotation matrices [batch, num_projections, 2, 2]
            rotations = self._generate_2d_rotations(batch_size, num_projections)
        
        # Random shifts [batch, num_projections, 2]
        shifts = torch.randn(batch_size, num_projections, 2, device=self.device) * 5.0
        
        return reconstructions, rotations, shifts
    
    def _generate_2d_rotations(self, batch_size: int, num_projections: int) -> torch.Tensor:
        """Generate random 2D rotation matrices."""
        angles = torch.rand(batch_size, num_projections, device=self.device) * 2 * torch.pi
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        
        rotations = torch.zeros(batch_size, num_projections, 2, 2, device=self.device)
        rotations[:, :, 0, 0] = cos_a
        rotations[:, :, 0, 1] = -sin_a
        rotations[:, :, 1, 0] = sin_a
        rotations[:, :, 1, 1] = cos_a
        
        return rotations
    
    def _generate_3d_rotations(self, batch_size: int, num_projections: int) -> torch.Tensor:
        """Generate random 3D rotation matrices using Euler angles."""
        import math
        
        angles_x = torch.rand(batch_size, num_projections, device=self.device) * 2 * math.pi
        angles_y = torch.rand(batch_size, num_projections, device=self.device) * 2 * math.pi
        angles_z = torch.rand(batch_size, num_projections, device=self.device) * 2 * math.pi
        
        rotations = torch.zeros(batch_size, num_projections, 3, 3, device=self.device)
        
        for i in range(batch_size):
            for j in range(num_projections):
                cos_x, sin_x = torch.cos(angles_x[i, j]), torch.sin(angles_x[i, j])
                cos_y, sin_y = torch.cos(angles_y[i, j]), torch.sin(angles_y[i, j])
                cos_z, sin_z = torch.cos(angles_z[i, j]), torch.sin(angles_z[i, j])
                
                # Individual rotation matrices
                Rx = torch.tensor([
                    [1, 0, 0],
                    [0, cos_x, -sin_x],
                    [0, sin_x, cos_x]
                ], dtype=torch.float32, device=self.device)
                
                Ry = torch.tensor([
                    [cos_y, 0, sin_y],
                    [0, 1, 0],
                    [-sin_y, 0, cos_y]
                ], dtype=torch.float32, device=self.device)
                
                Rz = torch.tensor([
                    [cos_z, -sin_z, 0],
                    [sin_z, cos_z, 0],
                    [0, 0, 1]
                ], dtype=torch.float32, device=self.device)
                
                # Combined rotation: Rz @ Ry @ Rx
                rotations[i, j] = Rz @ Ry @ Rx
        
        return rotations.to(self.device)
    
    def time_function(self, func, *args, **kwargs) -> Tuple[List[float], Dict[str, float]]:
        """Time a function with proper warmup and statistical analysis."""
        times = []
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            self._synchronize()
            func(*args, **kwargs)
        
        # Timing runs
        for _ in range(self.timing_runs):
            self._synchronize()
            start_time = time.perf_counter()
            func(*args, **kwargs)
            self._synchronize()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Remove outliers
        times = self._remove_outliers(times)
        
        # Calculate statistics
        stats = {
            "mean_time": statistics.mean(times),
            "median_time": statistics.median(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min_time": min(times),
            "max_time": max(times)
        }
        
        return times, stats
    
    def _synchronize(self):
        """Synchronize device operations."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()
    
    def _remove_outliers(self, times: List[float]) -> List[float]:
        """Remove statistical outliers from timing data."""
        if len(times) < 3:
            return times
        
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times)
        threshold = self.outlier_threshold * std_time
        
        filtered_times = [t for t in times if abs(t - mean_time) <= threshold]
        return filtered_times if filtered_times else times  # Keep original if all filtered out
    
    def measure_memory(self) -> Dict[str, float]:
        """Measure current memory usage."""
        memory_info = {}
        
        if self.device.type == "cuda":
            memory_info["gpu_memory_mb"] = torch.cuda.memory_allocated(self.device) / 1e6
            memory_info["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(self.device) / 1e6
        
        # System memory
        process = psutil.Process()
        memory_info["system_memory_mb"] = process.memory_info().rss / 1e6
        
        return memory_info
    
    def reset_memory_stats(self):
        """Reset memory statistics."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage since last reset.
        
        Note: Peak memory tracking is only available for CUDA devices.
        For MPS and CPU, no equivalent to torch.cuda.max_memory_allocated() exists.
        """
        memory_info = {}
        
        if self.device.type == "cuda":
            memory_info["gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated(self.device) / 1e6
        
        return memory_info
    
    def add_benchmark_result(self, test_name: str, parameters: Dict[str, Any], 
                           times: List[float], stats: Dict[str, float], 
                           additional_metrics: Optional[Dict[str, Any]] = None):
        """Add a benchmark result to the results structure."""
        result = {
            "parameters": parameters,
            "results": {
                **stats,
                "raw_times": times
            }
        }
        
        if additional_metrics:
            result["results"].update(additional_metrics)
        
        self.results["benchmarks"][test_name] = result
    
    def save_results(self):
        """Save results to JSON file."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {self.results_file}")
    
    def load_results(self) -> Dict[str, Any]:
        """Load existing results from JSON file."""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return {"metadata": {}, "benchmarks": {}}
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        print(f"\n{'='*60}")
        print(f"BENCHMARK SUMMARY: {self.experiment_name}")
        print(f"Platform: {self.platform_name}")
        print(f"Device: {self.device}")
        print(f"{'='*60}")
        
        for test_name, result in self.results["benchmarks"].items():
            results = result["results"]
            params = result["parameters"]
            
            print(f"\n{test_name}:")
            print(f"  Parameters: {params}")
            print(f"  Median time: {results['median_time']:.4f} seconds")
            print(f"  Mean time: {results['mean_time']:.4f} ± {results['std_dev']:.4f} seconds")
            
            if "throughput_proj_per_sec" in results:
                print(f"  Throughput: {results['throughput_proj_per_sec']:.1f} projections/sec")
            
            if "gpu_peak_memory_mb" in results:
                print(f"  Peak GPU memory: {results['gpu_peak_memory_mb']:.1f} MB")


def create_argument_parser(description: str) -> argparse.ArgumentParser:
    """Create a common argument parser for benchmark scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--platform-name', 
        required=True,
        help='Platform identifier (e.g., "a100-cuda", "m2-mps", "intel-cpu")'
    )
    parser.add_argument(
        '--warmup-runs',
        type=int,
        default=5,
        help='Number of warmup runs (default: 5)'
    )
    parser.add_argument(
        '--timing-runs',
        type=int,
        default=20,
        help='Number of timing runs (default: 20)'
    )
    parser.add_argument(
        '--batch-sizes',
        type=int,
        nargs='+',
        default=[1, 8, 16, 32],
        help='Batch sizes to test (default: 1 8 16 32)'
    )
    parser.add_argument(
        '--image-sizes',
        type=int,
        nargs='+',
        default=[64, 128, 256],
        help='Image sizes to test (default: 64 128 256)'
    )
    return parser