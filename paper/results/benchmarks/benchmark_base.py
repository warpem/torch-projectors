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
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse

import torch
import psutil


class BenchmarkBase:
    """Base class for all benchmarking scripts."""
    
    @staticmethod
    def format_value_safe(value, decimal_places=4, unit="", none_replacement="N/A"):
        """Safely format a numeric value, handling None values."""
        if value is None:
            return none_replacement
        return f"{value:.{decimal_places}f}{unit}"
    
    def __init__(self, platform_name: str, experiment_name: str, device: Optional[torch.device] = None, title: Optional[str] = None):
        self.platform_name = platform_name
        self.experiment_name = experiment_name
        self.device = device if device is not None else self._detect_device()
        self.title = title
        self.results_dir = Path(__file__).parent.parent / "data" / experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.results_dir / f"{platform_name}.json"
        
        # Benchmark configuration
        self.warmup_runs = 5
        self.timing_runs = 20
        self.cooldown_seconds = 0.0  # Cooldown between runs
        self.test_cooldown_seconds = 5.0  # Cooldown after each test matrix element
        self.profile_memory = False  # Memory profiling mode
        
        # Subprocess execution mode
        self.use_subprocess_isolation = True  # Enable by default for memory isolation
        
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
        
        if self.title:
            metadata["title"] = self.title
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
                dtype=torch.complex64, device='cpu'
            ).to(self.device)
            # 3D rotation matrices [batch, num_projections, 3, 3]
            rotations = self._generate_3d_rotations(batch_size, num_projections)
        else:
            # 2D reconstructions in RFFT format: [batch, height, width//2 + 1]  
            reconstructions = torch.randn(
                batch_size, height, width // 2 + 1,
                dtype=torch.complex64, device='cpu'
            ).to(self.device)
            # 2D rotation matrices [batch, num_projections, 2, 2]
            rotations = self._generate_2d_rotations(batch_size, num_projections)
        
        # Random shifts [batch, num_projections, 2]
        shifts = torch.randn(batch_size, num_projections, 2, device='cpu').to(self.device) * 5.0
        
        return reconstructions, rotations, shifts
    
    def _generate_2d_rotations(self, batch_size: int, num_projections: int) -> torch.Tensor:
        """Generate random 2D rotation matrices."""
        angles = torch.rand(batch_size, num_projections, device='cpu').to(self.device) * 2 * torch.pi
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
        
        angles_x = torch.rand(batch_size, num_projections, device='cpu').to(self.device) * 2 * math.pi
        angles_y = torch.rand(batch_size, num_projections, device='cpu').to(self.device) * 2 * math.pi
        angles_z = torch.rand(batch_size, num_projections, device='cpu').to(self.device) * 2 * math.pi
        
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
        for i in range(self.warmup_runs):
            self._synchronize()
            func(*args, **kwargs)
            if i < self.warmup_runs - 1 and self.cooldown_seconds > 0:
                time.sleep(self.cooldown_seconds)
        
        # Cooldown after warmup
        if self.cooldown_seconds > 0:
            time.sleep(self.cooldown_seconds)
        
        # Timing runs
        for i in range(self.timing_runs):
            # Wake up CPU from power management before measurement
            # self._wakeup_cpu()
            
            self._synchronize()
            start_time = time.perf_counter()
            func(*args, **kwargs)
            self._synchronize()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            
            # Cooldown between timing runs (except after the last run)
            if i < self.timing_runs - 1 and self.cooldown_seconds > 0:
                time.sleep(self.cooldown_seconds)
        
        # Calculate statistics (using median as primary metric)
        stats = {
            "median_time": statistics.median(times),
            "mean_time": statistics.mean(times),
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
    
    def _wakeup_cpu(self):
        """Perform a small intensive operation to wake up CPU from power management."""
        # Small matrix multiplication to activate CPU cores
        a = torch.randn(64, 64, device=self.device, dtype=torch.float32)
        b = torch.randn(64, 64, device=self.device, dtype=torch.float32)
        _ = torch.mm(a, b)
        self._synchronize()
    
    
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
    
    def profile_memory_usage_detailed(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Profile memory usage with memory_profiler for detailed analysis.
        
        This method uses memory_profiler to capture fine-grained memory usage over time.
        Uses the memory timeline itself as both baseline and peak measurements.
        """
        from memory_profiler import memory_usage
        import gc
        
        # Force garbage collection before measurement to get clean baseline
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        self.reset_memory_stats()
        
        # Wrapper function to capture memory usage over time
        def memory_monitored_function():
            self._synchronize()
            result = func(*args, **kwargs)
            self._synchronize()
            return result
        
        # Monitor memory usage during function execution
        # Sample every 1ms for good resolution
        mem_usage = memory_usage(
            (memory_monitored_function, (), {}),
            interval=0.001,  # 1ms intervals
            timeout=None,
            include_children=False,
            multiprocess=False,
            max_usage=False,
            retval=True,
            backend='psutil'
        )
        
        if isinstance(mem_usage, tuple):
            memory_timeline, result = mem_usage
        else:
            memory_timeline = mem_usage
            result = None
        
        # Get PyTorch-specific peak memory if available
        peak_memory = self.get_peak_memory()
        
        # Calculate memory statistics from timeline
        if memory_timeline and len(memory_timeline) > 0:
            # Use first measurement as baseline, max as peak
            baseline_mb = memory_timeline[0] 
            peak_mb = max(memory_timeline)
            min_mb = min(memory_timeline)
            avg_mb = sum(memory_timeline) / len(memory_timeline)
            
            memory_profile = {
                "baseline_system_memory_mb": baseline_mb,
                "peak_system_memory_mb": peak_mb,
                "min_system_memory_mb": min_mb,
                "avg_system_memory_mb": avg_mb,
                "system_memory_delta_mb": peak_mb - baseline_mb,
                "memory_timeline_mb": memory_timeline[:50],  # Limit stored samples
                "timeline_length": len(memory_timeline),
                "sampling_interval_ms": 1.0
            }
        else:
            # Fallback to basic measurement if timeline failed
            baseline_memory = self.measure_memory()
            memory_profile = {
                "baseline_system_memory_mb": baseline_memory.get("system_memory_mb", 0.0),
                "peak_system_memory_mb": baseline_memory.get("system_memory_mb", 0.0),
                "system_memory_delta_mb": 0.0
            }
        
        # Add CUDA memory stats if available
        if self.device.type == "cuda":
            baseline_gpu = self.measure_memory().get("gpu_memory_mb", 0.0)
            memory_profile.update({
                "baseline_gpu_memory_mb": baseline_gpu,
                "peak_gpu_memory_mb": peak_memory.get("gpu_peak_memory_mb", baseline_gpu),
                "gpu_memory_delta_mb": peak_memory.get("gpu_peak_memory_mb", baseline_gpu) - baseline_gpu
            })
        
        # Clean up result to free memory
        if result:
            del result
        
        return memory_profile
    
    def calculate_tensor_sizes(self, *tensors, include_gradients=False) -> Dict[str, Any]:
        """Calculate the size of tensors in MB by directly measuring them.
        
        Args:
            *tensors: Variable number of tensors to measure
            include_gradients: Whether to include gradient tensors in the calculation
        """
        total_bytes = 0
        gradient_bytes = 0
        tensor_info = {}
        
        for i, tensor in enumerate(tensors):
            if tensor is not None:
                # Calculate bytes: num_elements * element_size
                num_elements = tensor.numel()
                element_size = tensor.element_size()
                tensor_bytes = num_elements * element_size
                tensor_mb = tensor_bytes / 1e6
                
                total_bytes += tensor_bytes
                tensor_info[f"tensor_{i}_mb"] = tensor_mb
                tensor_info[f"tensor_{i}_shape"] = list(tensor.shape)
                tensor_info[f"tensor_{i}_dtype"] = str(tensor.dtype)
                tensor_info[f"tensor_{i}_requires_grad"] = tensor.requires_grad
                
                # Include gradient tensors if requested and they exist
                # Only check gradients on leaf tensors to avoid warnings
                if include_gradients and tensor.is_leaf and tensor.grad is not None:
                    grad_bytes = tensor.grad.numel() * tensor.grad.element_size()
                    grad_mb = grad_bytes / 1e6
                    gradient_bytes += grad_bytes
                    tensor_info[f"tensor_{i}_grad_mb"] = grad_mb
                    tensor_info[f"tensor_{i}_grad_shape"] = list(tensor.grad.shape)
                    tensor_info[f"tensor_{i}_is_leaf"] = True
                elif include_gradients:
                    tensor_info[f"tensor_{i}_is_leaf"] = tensor.is_leaf
        
        tensor_info["total_mb"] = total_bytes / 1e6
        if include_gradients:
            tensor_info["total_gradients_mb"] = gradient_bytes / 1e6
            tensor_info["total_with_gradients_mb"] = (total_bytes + gradient_bytes) / 1e6
        
        return tensor_info
    
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
            # Handle nested results structure
            if "forward" in results:
                print(f"  Forward median time: {self.format_value_safe(results['forward']['median_time'], 4, ' seconds')}")
                if "backward" in results:
                    print(f"  Backward median time: {self.format_value_safe(results['backward']['median_time'], 4, ' seconds')}")
                if "forward_and_backward" in results:
                    print(f"  Forward+Backward median time: {self.format_value_safe(results['forward_and_backward']['median_time'], 4, ' seconds')}")
            elif "forward_no_grad" in results:
                print(f"  Forward (no-grad) median time: {self.format_value_safe(results['forward_no_grad']['median_time'], 4, ' seconds')}")
            elif "median_time" in results:
                print(f"  Median time: {self.format_value_safe(results['median_time'], 4, ' seconds')}")
                mean_time = self.format_value_safe(results['mean_time'], 4)
                std_dev = self.format_value_safe(results['std_dev'], 4)
                print(f"  Mean time: {mean_time} ± {std_dev} seconds")
            
            if "throughput_proj_per_sec" in results:
                print(f"  Throughput: {self.format_value_safe(results['throughput_proj_per_sec'], 1, ' projections/sec')}")
            
            # Check for error information
            if "error" in results:
                print(f"  ERROR: {results['error']}")
            
            if "gpu_peak_memory_mb" in results:
                print(f"  Peak GPU memory: {self.format_value_safe(results['gpu_peak_memory_mb'], 1, ' MB')}")
            
            # Memory profiling results
            if "memory_profile" in results:
                mem_profile = results["memory_profile"]
                print(f"  Memory Profile:")
                
                if "forward_memory" in mem_profile:
                    fm = mem_profile["forward_memory"]
                    if "peak_gpu_memory_mb" in fm:
                        print(f"    Forward GPU memory: {self.format_value_safe(fm['peak_gpu_memory_mb'], 1, ' MB')}")
                    elif "system_memory_delta_mb" in fm and fm["system_memory_delta_mb"] is not None and abs(fm["system_memory_delta_mb"]) > 0.1:
                        print(f"    Forward memory delta: {self.format_value_safe(fm['system_memory_delta_mb'], 1, ' MB')}")
                    
                if "backward_memory" in mem_profile:
                    bm = mem_profile["backward_memory"] 
                    if "peak_gpu_memory_mb" in bm:
                        print(f"    Backward GPU memory: {self.format_value_safe(bm['peak_gpu_memory_mb'], 1, ' MB')}")
                    elif "system_memory_delta_mb" in bm and bm["system_memory_delta_mb"] is not None and abs(bm["system_memory_delta_mb"]) > 0.1:
                        print(f"    Backward memory delta: {self.format_value_safe(bm['system_memory_delta_mb'], 1, ' MB')}")
                
                if "input_data_sizes" in mem_profile:
                    input_sizes = mem_profile["input_data_sizes"]
                    input_size = input_sizes["total_mb"]
                    print(f"    Input data size: {input_size:.1f} MB")
                    
                    if "total_gradients_mb" in input_sizes and input_sizes["total_gradients_mb"] > 0:
                        grad_size = input_sizes["total_gradients_mb"]
                        print(f"    Input gradient size: {grad_size:.1f} MB")
                    
                if "output_data_sizes" in mem_profile:
                    output_size = mem_profile["output_data_sizes"]["total_mb"]
                    print(f"    Output data size: {output_size:.1f} MB")
                
                if "gradient_data_sizes" in mem_profile:
                    grad_sizes = mem_profile["gradient_data_sizes"]
                    if "total_gradients_mb" in grad_sizes and grad_sizes["total_gradients_mb"] > 0:
                        grad_size = grad_sizes["total_gradients_mb"]
                        total_size = grad_sizes["total_with_gradients_mb"]
                        print(f"    Expected gradient size: {grad_size:.1f} MB")
                        print(f"    Total with gradients: {total_size:.1f} MB")
                    
                if "io_ratio" in mem_profile:
                    ratio = mem_profile["io_ratio"]
                    print(f"    Output/Input ratio: {ratio:.1f}x")
                    
                if "forward_memory_efficiency" in mem_profile:
                    efficiency = mem_profile["forward_memory_efficiency"]
                    print(f"    Memory efficiency: {efficiency:.1f}x input size")
    
    def run_single_element_subprocess(self, script_path: str, element_params: Dict[str, Any], 
                                    temp_dir: Path) -> Optional[Dict[str, Any]]:
        """Run a single benchmark element in a subprocess with memory isolation.
        
        Args:
            script_path: Path to the benchmark script to execute
            element_params: Dictionary of parameters for this specific element
            temp_dir: Directory for temporary result files
            
        Returns:
            Dictionary containing the benchmark results, or None if failed
        """
        import shutil
        
        # Create unique temporary file for this element's results
        result_file = temp_dir / f"element_{uuid.uuid4().hex}.json"
        
        # Build subprocess command
        cmd = [
            sys.executable, script_path,
            "--platform-name", self.platform_name,
            "--device", str(self.device),
            "--warmup-runs", str(self.warmup_runs),
            "--timing-runs", str(self.timing_runs),
            "--cooldown", str(self.cooldown_seconds),
            "--single-element", str(result_file)
        ]
        
        # Add element-specific parameters
        for key, value in element_params.items():
            if key == "batch_size":
                cmd.extend(["--batch-sizes", str(value)])
            elif key == "image_size":
                cmd.extend(["--image-sizes", str(value)])
            elif key == "num_projections":
                cmd.extend(["--num-projections", str(value)])
            elif key == "interpolation":
                cmd.extend(["--interpolations", str(value)])
        
        if self.profile_memory:
            cmd.append("--profile-memory")
            
        if self.title:
            cmd.extend(["--title", self.title])
        
        try:
            print(f"Running subprocess for {element_params}...")
            
            # Run subprocess with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per element
                cwd=Path(script_path).parent
            )
            
            if result.returncode != 0:
                print(f"Subprocess failed for {element_params}")
                print(f"STDERR: {result.stderr}")
                print(f"STDOUT: {result.stdout}")
                return None
            
            # Read results from temporary file
            if result_file.exists():
                with open(result_file, 'r') as f:
                    element_result = json.load(f)
                result_file.unlink()  # Clean up temp file
                
                # Wait for test cooldown
                if self.test_cooldown_seconds > 0:
                    print(f"Cooling down for {self.test_cooldown_seconds}s...")
                    time.sleep(self.test_cooldown_seconds)
                
                return element_result
            else:
                print(f"Result file not found for {element_params}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"Subprocess timed out for {element_params}")
            return None
        except Exception as e:
            print(f"Error running subprocess for {element_params}: {e}")
            return None
    
    def run_matrix_with_subprocess_isolation(self, script_path: str, 
                                           parameter_matrix: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the entire benchmark matrix using subprocess isolation.
        
        Each matrix element runs in a separate process for complete memory isolation.
        
        Args:
            script_path: Path to the benchmark script
            parameter_matrix: List of parameter combinations to test
            
        Returns:
            Combined results from all elements
        """
        print(f"Running {len(parameter_matrix)} benchmark elements with subprocess isolation...")
        
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            combined_results = {
                "metadata": self._gather_system_info(),
                "benchmarks": {}
            }
            
            successful_elements = 0
            
            for i, element_params in enumerate(parameter_matrix):
                print(f"\n[{i+1}/{len(parameter_matrix)}] Testing {element_params}")
                
                element_result = self.run_single_element_subprocess(
                    script_path, element_params, temp_dir
                )
                
                if element_result is not None:
                    # Extract the benchmark result (should be single element)
                    for test_name, result_data in element_result.get("benchmarks", {}).items():
                        combined_results["benchmarks"][test_name] = result_data
                    successful_elements += 1
                else:
                    # Create error entry
                    test_name = self._generate_test_name(element_params)
                    combined_results["benchmarks"][test_name] = {
                        "parameters": element_params,
                        "results": {"error": "Subprocess execution failed"}
                    }
            
            print(f"\nCompleted: {successful_elements}/{len(parameter_matrix)} elements successful")
            return combined_results
    
    def _generate_test_name(self, params: Dict[str, Any]) -> str:
        """Generate a test name from parameters."""
        parts = []
        for key in sorted(params.keys()):
            parts.append(f"{key}_{params[key]}")
        return "_".join(parts)
    
    @classmethod
    def run_single_element_mode(cls, args, benchmark_function):
        """Run benchmark in single-element mode for subprocess execution.
        
        This method should be called by benchmark scripts when --single-element is specified.
        
        Args:
            args: Parsed command line arguments
            benchmark_function: Function that runs the actual benchmark
        """
        if not hasattr(args, 'single_element') or not args.single_element:
            raise ValueError("Single element mode requires --single-element argument")
        
        result_file = Path(args.single_element)
        
        # Create benchmark instance - use factory method if available, otherwise try standard constructor
        device = cls.parse_device(args.device)
        try:
            # Try the subclass constructor (might have different signature)
            benchmark = cls(args.platform_name, device, args.title)
        except TypeError:
            # Fallback to base class constructor with default experiment name
            benchmark = cls(args.platform_name, "single_element", device, args.title)
        
        # Apply configuration from args
        benchmark.warmup_runs = args.warmup_runs
        benchmark.timing_runs = args.timing_runs
        benchmark.cooldown_seconds = args.cooldown
        benchmark.profile_memory = args.profile_memory
        
        # Run the benchmark function (should add results to benchmark.results)
        try:
            benchmark_function(benchmark, args)
            
            # Save results to temporary file
            with open(result_file, 'w') as f:
                json.dump(benchmark.results, f, indent=2)
                
        except Exception as e:
            # Save error information
            error_result = {
                "metadata": benchmark._gather_system_info(),
                "benchmarks": {
                    "error": {
                        "parameters": {},
                        "results": {"error": str(e)}
                    }
                }
            }
            with open(result_file, 'w') as f:
                json.dump(error_result, f, indent=2)
            raise
    
    @staticmethod
    def calculate_thermal_load(batch_size: int, image_size: int, num_projections: int, 
                              interpolation: str = 'linear') -> float:
        """Calculate thermal load metric for a test configuration.
        
        Load correlates with batch_size * num_projections * image_size^2 * interpolation_factor
        since operations scale quadratically with image dimensions, and cubic interpolation
        is approximately 4x more computationally intensive than linear.
        
        Args:
            batch_size: Number of reconstructions in batch
            image_size: Size of square image (height/width)
            num_projections: Number of projections
            interpolation: 'linear' or 'cubic'
        """
        base_load = float(batch_size * num_projections * image_size * image_size)
        
        # Apply interpolation complexity multiplier
        interpolation_factor = 4.0 if interpolation == 'cubic' else 1.0
        
        return base_load * interpolation_factor
    
    @staticmethod
    def create_zigzag_ordering(parameter_matrix: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create zigzag ordering that alternates between light and heavy tests.
        
        This maximizes cooling time between heavy tests by ensuring they're
        always separated by lighter tests.
        
        Args:
            parameter_matrix: List of parameter dicts, each containing test config
            
        Returns:
            Reordered parameter matrix with zigzag light/heavy pattern
        """
        # Calculate thermal load for each test and sort
        tests_with_load = []
        for params in parameter_matrix:
            # Use the calculate_thermal_load method if parameters are available
            if all(key in params for key in ["batch_size", "image_size", "num_projections"]):
                load = BenchmarkBase.calculate_thermal_load(
                    params["batch_size"], params["image_size"], 
                    params["num_projections"], params.get("interpolation", "linear")
                )
            else:
                # Fallback to simple heuristic based on available parameters
                load = 1.0
                if "batch_size" in params:
                    load *= params["batch_size"]
                if "image_size" in params:
                    load *= params["image_size"] ** 2
                if "num_projections" in params:
                    load *= params["num_projections"]
            
            tests_with_load.append((load, params))
        
        # Sort by thermal load (ascending)
        tests_with_load.sort(key=lambda x: x[0])
        
        # Create zigzag ordering: alternate from front (light) and back (heavy)
        zigzag_ordered = []
        front_idx = 0
        back_idx = len(tests_with_load) - 1
        take_from_front = True
        
        while front_idx <= back_idx:
            if take_from_front:
                zigzag_ordered.append(tests_with_load[front_idx][1])
                front_idx += 1
            else:
                zigzag_ordered.append(tests_with_load[back_idx][1])
                back_idx -= 1
            take_from_front = not take_from_front
        
        return zigzag_ordered
    
    def profile_forward_backward_memory(self, forward_func, backward_func = None, 
                                      input_tensors = None, output_sample_func = None) -> Dict[str, Any]:
        """Profile memory usage for forward and optional backward operations.
        
        Args:
            forward_func: Function that performs forward operation
            backward_func: Optional function that performs forward+backward operation
            input_tensors: Optional tuple of input tensors for size calculation
            output_sample_func: Optional function that returns a sample output tensor
            
        Returns:
            Dictionary containing memory profiling information
        """
        memory_profile = {}
        
        if not self.profile_memory:
            return memory_profile
            
        print(f"      Profiling memory usage...")
        
        # Profile forward pass with detailed memory tracking
        forward_memory = self.profile_memory_usage_detailed(forward_func)
        memory_profile["forward_memory"] = forward_memory
        
        # Profile backward pass if provided
        if backward_func is not None:
            backward_memory = self.profile_memory_usage_detailed(backward_func)
            memory_profile["backward_memory"] = backward_memory
        
        # Calculate input tensor sizes if provided
        if input_tensors is not None:
            input_sizes = self.calculate_tensor_sizes(*input_tensors, include_gradients=True)
            memory_profile["input_data_sizes"] = input_sizes
        
        # Calculate output tensor sizes if sample function provided
        if output_sample_func is not None:
            with torch.no_grad():
                sample_output = output_sample_func()
            output_sizes = self.calculate_tensor_sizes(sample_output, include_gradients=False)
            memory_profile["output_data_sizes"] = output_sizes
            del sample_output
        
        return memory_profile
    
    def measure_gradient_sizes(self, forward_func, input_tensors, loss_func = None) -> Dict[str, Any]:
        """Measure the expected gradient sizes after backward pass.
        
        Args:
            forward_func: Function that performs forward operation
            input_tensors: Tuple of input tensors  
            loss_func: Optional function to compute loss from forward output
            
        Returns:
            Dictionary containing gradient size information
        """
        if not input_tensors:
            return {}
            
        # Use minimal sample to measure gradient creation
        test_tensors = []
        for tensor in input_tensors:
            if tensor is not None and tensor.numel() > 0:
                # Use first element of first batch for minimal test
                if tensor.ndim >= 2:
                    test_tensor = tensor[:1].clone().requires_grad_(True)
                else:
                    test_tensor = tensor.clone().requires_grad_(True)
                test_tensors.append(test_tensor)
            else:
                test_tensors.append(tensor)
        
        try:
            # Run minimal forward+backward to create gradients
            output = forward_func(*test_tensors)
            if loss_func is not None:
                loss = loss_func(output)
            else:
                # Default loss function
                if hasattr(output, 'sum'):
                    loss = torch.sum(torch.abs(output)**2)
                else:
                    loss = output  # Assume it's already a scalar loss
            
            loss.backward()
            
            # Measure gradient sizes
            gradient_sizes = self.calculate_tensor_sizes(*test_tensors, include_gradients=True)
            
            # Clean up test tensors
            for tensor in test_tensors:
                if tensor is not None and hasattr(tensor, 'grad') and tensor.grad is not None:
                    tensor.grad = None
            del test_tensors, output, loss
            
            return gradient_sizes
            
        except Exception as e:
            print(f"Warning: Could not measure gradient sizes: {e}")
            return {}
    
    def calculate_memory_efficiency_metrics(self, memory_profile: Dict[str, Any], 
                                          total_projections: Optional[int] = None) -> Dict[str, float]:
        """Calculate memory efficiency metrics from memory profile data.
        
        Args:
            memory_profile: Memory profile data from profile_forward_backward_memory
            total_projections: Total number of projections for per-projection metrics
            
        Returns:
            Dictionary containing efficiency metrics
        """
        metrics = {}
        
        # Extract memory data
        forward_memory = memory_profile.get("forward_memory", {})
        input_sizes = memory_profile.get("input_data_sizes", {})
        output_sizes = memory_profile.get("output_data_sizes", {})
        
        # Calculate memory per projection if total_projections provided
        if total_projections and total_projections > 0:
            if "peak_gpu_memory_mb" in forward_memory:
                metrics["forward_memory_per_proj_mb"] = forward_memory["peak_gpu_memory_mb"] / total_projections
            elif "peak_system_memory_mb" in forward_memory:
                metrics["forward_memory_per_proj_mb"] = forward_memory["peak_system_memory_mb"] / total_projections
        
        # Calculate memory efficiency (peak memory / input size)  
        input_size_mb = input_sizes.get("total_mb", 0.0)
        if input_size_mb > 0:
            if "peak_gpu_memory_mb" in forward_memory:
                metrics["forward_memory_efficiency"] = forward_memory["peak_gpu_memory_mb"] / input_size_mb
            elif "system_memory_delta_mb" in forward_memory:
                metrics["forward_memory_efficiency"] = forward_memory["system_memory_delta_mb"] / input_size_mb
        
        # Calculate I/O ratio (output size / input size)
        output_size_mb = output_sizes.get("total_mb", 0.0)
        if input_size_mb > 0:
            metrics["io_ratio"] = output_size_mb / input_size_mb
        
        return metrics
    
    def time_forward_backward_operations(self, forward_func, forward_backward_func) -> Dict[str, Any]:
        """Time forward and forward+backward operations, calculating pure backward time.
        
        Args:
            forward_func: Function that performs only forward operation
            forward_backward_func: Function that performs forward+backward operation
            
        Returns:
            Dictionary containing timing statistics for forward, backward, and combined operations
        """
        # Reset memory stats
        self.reset_memory_stats()
        
        # Time forward pass
        forward_times, forward_stats = self.time_function(forward_func)
        
        # Time backward pass (forward + backward)
        backward_times, backward_stats = self.time_function(forward_backward_func)
        
        # Calculate pure backward time by subtracting forward time
        pure_backward_times = []
        for b_time, f_time in zip(backward_times, forward_times):
            pure_backward_times.append(max(0.0, b_time - f_time))
            
        pure_backward_stats = {
            "median_time": pure_backward_times[len(pure_backward_times)//2] if pure_backward_times else 0.0,
            "mean_time": sum(pure_backward_times) / len(pure_backward_times) if pure_backward_times else 0.0,
            "std_dev": 0.0,  # Simplified for now
            "min_time": min(pure_backward_times) if pure_backward_times else 0.0,
            "max_time": max(pure_backward_times) if pure_backward_times else 0.0
        }
        
        # Get peak memory usage
        peak_memory = self.get_peak_memory()
        
        return {
            "forward": forward_stats,
            "backward": pure_backward_stats,
            "forward_and_backward": backward_stats,
            **peak_memory
        }
    
    def time_operation_with_throughput(self, operation_func, total_items: int) -> Dict[str, Any]:
        """Time an operation and calculate throughput.
        
        Args:
            operation_func: Function to time
            total_items: Total number of items processed (for throughput calculation)
            
        Returns:
            Dictionary containing timing stats and throughput
        """
        # Reset memory stats
        self.reset_memory_stats()
        
        # Time operation
        _, stats = self.time_function(operation_func)
        
        # Get peak memory usage
        peak_memory = self.get_peak_memory()
        
        # Calculate throughput
        throughput = total_items / stats["median_time"] if stats["median_time"] > 0 else 0.0
        
        return {
            **stats,
            "throughput_items_per_sec": throughput,
            **peak_memory
        }
    
    def create_gradient_functions(self, base_func, input_tensors, loss_func=None):
        """Create forward and forward+backward functions for gradient timing.
        
        Args:
            base_func: Base function that performs the operation
            input_tensors: Input tensors for the operation
            loss_func: Optional loss function (defaults to sum of squared absolute values)
            
        Returns:
            Tuple of (forward_func, forward_backward_func)
        """
        def forward_pass():
            """Forward operation without gradients."""
            # Set requires_grad on inputs
            for tensor in input_tensors:
                if tensor is not None and tensor.is_floating_point():
                    tensor.requires_grad_(True)
            
            result = base_func(*input_tensors)
            # Don't return tensor to avoid memory profiler issues
            del result
            
        def backward_pass():
            """Combined forward + backward pass."""
            # Set requires_grad on inputs
            for tensor in input_tensors:
                if tensor is not None and tensor.is_floating_point():
                    tensor.requires_grad_(True)
            
            result = base_func(*input_tensors)
            
            if loss_func is not None:
                loss = loss_func(result)
            else:
                # Default loss: sum of squared absolute values
                if hasattr(result, 'sum'):
                    loss = torch.sum(torch.abs(result)**2)
                else:
                    loss = result  # Assume it's already a scalar
            
            loss.backward()
            
            # Clear gradients to avoid accumulation
            for tensor in input_tensors:
                if tensor is not None and hasattr(tensor, 'grad') and tensor.grad is not None:
                    tensor.grad = None
            
            del result, loss
            
        return forward_pass, backward_pass


def create_argument_parser(description: str) -> argparse.ArgumentParser:
    """Create a common argument parser for benchmark scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--platform-name', 
        required=True,
        help='Platform identifier (e.g., "a100-cuda", "m2-mps", "intel-cpu")'
    )
    parser.add_argument(
        '--title',
        type=str,
        help='Custom title for the benchmark results table'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use: auto, cpu, cuda, mps (default: auto)'
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
        default=[1, 8],
        help='Batch sizes to test (default: 1 8)'
    )
    parser.add_argument(
        '--image-sizes',
        type=int,
        nargs='+',
        default=[32, 128, 256],
        help='Image sizes to test (default: 32 128 256)'
    )
    parser.add_argument(
        '--num-projections',
        type=int,
        nargs='+',
        default=[8, 128, 2048],
        help='Number of projections per batch (default: 8 128 2048)'
    )
    parser.add_argument(
        '--interpolations',
        type=str,
        nargs='+',
        default=['linear', 'cubic'],
        help='Interpolation methods to test (default: linear cubic)'
    )
    parser.add_argument(
        '--cooldown',
        type=float,
        default=0.0,
        help='Cooldown time in seconds between runs to prevent thermal throttling (default: 0.0)'
    )
    parser.add_argument(
        '--test-cooldown',
        type=float,
        default=10.0,
        help='Cooldown time in seconds after each test matrix element (default: 60.0)'
    )
    parser.add_argument(
        '--profile-memory',
        action='store_true',
        help='Run additional memory profiling (separate from timing runs)'
    )
    parser.add_argument(
        '--single-element',
        type=str,
        help='Run single benchmark element and save results to specified file (for subprocess isolation)'
    )
    return parser