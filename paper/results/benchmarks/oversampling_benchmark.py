#!/usr/bin/env python3
"""
Benchmark script for oversampling scenarios in 2D forward projection.

This script tests torch_projectors.project_2d_forw performance with different
oversampling configurations and padding strategies:
1. Basic 128x128 forward+backward with 4096 projections
2. 256x256 with oversampling=2 and 128x128 output
3. Real-space padding from 128x128 to 256x256, then oversampling=2

Usage:
    python oversampling_benchmark.py --platform-name "a100-cuda" --device auto
    python oversampling_benchmark.py --platform-name "m2-mps" --device mps --interpolations linear cubic
"""

import sys
import time
import math
import torch
import torch.nn.functional as F
import torch_projectors
from pathlib import Path
from typing import Optional

# Add benchmark_base to path  
sys.path.append(str(Path(__file__).parent))
from benchmark_base import BenchmarkBase, create_argument_parser


class OversamplingBenchmark(BenchmarkBase):
    """Benchmark class for oversampling scenarios."""
    
    def __init__(self, platform_name: str, device: Optional[torch.device] = None, title: Optional[str] = None):
        super().__init__(platform_name, "oversampling_benchmark", device, title)
        
        # Fixed parameters for all scenarios
        self.num_reconstructions = 4096
        self.num_projections_per_reconstruction = 1
        
    def generate_projection_poses(self):
        """Generate unique random poses for each of the 4096 reconstructions (1 projection each)."""
        torch.manual_seed(42)  # Fixed seed for reproducibility
        
        # Random angles - one per reconstruction
        angles = torch.rand(self.num_reconstructions, self.num_projections_per_reconstruction, device='cpu').to(self.device) * 2 * math.pi
        
        # Create rotation matrices
        rotations = torch.zeros(self.num_reconstructions, self.num_projections_per_reconstruction, 2, 2, device=self.device)
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        rotations[:, :, 0, 0] = cos_a
        rotations[:, :, 0, 1] = -sin_a
        rotations[:, :, 1, 0] = sin_a
        rotations[:, :, 1, 1] = cos_a
        
        # Random shifts - one per reconstruction
        shifts = torch.randn(self.num_reconstructions, self.num_projections_per_reconstruction, 2, device='cpu').to(self.device) * 5.0
        
        return rotations, shifts
    
    def benchmark_scenario_1(self, interpolation: str) -> dict:
        """
        Benchmark scenario 1: Basic 128x128 forward+backward.
        Start with 4096 different 128x(128/2+1) complex noise tensors, 1 projection each.
        """
        print(f"      Running scenario 1 (128x128 basic) with {interpolation} interpolation...")
        
        torch.manual_seed(42)
        
        # Generate 4096 different 128x65 complex tensors (RFFT format for 128x128 real)
        reconstruction = torch.randn(
            self.num_reconstructions, 128, 65, 
            dtype=torch.complex64, device=self.device
        )
        
        rotations, shifts = self.generate_projection_poses()
        
        def forward_pass():
            """Forward projection operation."""
            reconstruction.requires_grad_(True)
            projections = torch_projectors.project_2d_forw(
                reconstruction, rotations, shifts,
                output_shape=(128, 128),
                interpolation=interpolation
            )
            del projections
            
        def backward_pass():
            """Combined forward + backward pass."""
            reconstruction.requires_grad_(True)
            projections = torch_projectors.project_2d_forw(
                reconstruction, rotations, shifts,
                output_shape=(128, 128),
                interpolation=interpolation
            )
            loss = torch.sum(torch.abs(projections)**2)
            loss.backward()
            reconstruction.grad = None
            del projections, loss
        
        # Time both passes
        timing_results = self.time_forward_backward_operations(forward_pass, backward_pass)
        
        # Calculate throughput
        total_projections = self.num_reconstructions * self.num_projections_per_reconstruction
        forward_throughput = total_projections / timing_results["forward"]["median_time"] if timing_results["forward"]["median_time"] > 0 else 0.0
        total_throughput = total_projections / timing_results["forward_and_backward"]["median_time"] if timing_results["forward_and_backward"]["median_time"] > 0 else 0.0
        
        timing_results["throughput_forward_proj_per_sec"] = forward_throughput
        timing_results["throughput_total_proj_per_sec"] = total_throughput
        
        # Add memory profiling if enabled
        if self.profile_memory:
            memory_profile = self.profile_forward_backward_memory(forward_pass, backward_pass)
            timing_results["memory_profile"] = memory_profile
            
            # Add memory efficiency metrics
            if memory_profile:
                efficiency_metrics = self.calculate_memory_efficiency_metrics(memory_profile, total_projections)
                timing_results["memory_profile"].update(efficiency_metrics)
        
        return timing_results
    
    def benchmark_scenario_2(self, interpolation: str) -> dict:
        """
        Benchmark scenario 2: 256x256 with oversampling=2.
        Start with 4096 different 256x(256/2+1) complex noise tensors, use oversampling=2, output 128x128.
        """
        print(f"      Running scenario 2 (256x256 oversampling=2) with {interpolation} interpolation...")
        
        torch.manual_seed(42)
        
        # Generate 4096 different 256x129 complex tensors (RFFT format for 256x256 real)
        reconstruction = torch.randn(
            self.num_reconstructions, 256, 129, 
            dtype=torch.complex64, device=self.device
        )
        
        rotations, shifts = self.generate_projection_poses()
        
        def forward_pass():
            """Forward projection operation with oversampling=2."""
            reconstruction.requires_grad_(True)
            projections = torch_projectors.project_2d_forw(
                reconstruction, rotations, shifts,
                output_shape=(128, 128),
                interpolation=interpolation,
                oversampling=2.0
            )
            del projections
            
        def backward_pass():
            """Combined forward + backward pass with oversampling=2."""
            reconstruction.requires_grad_(True)
            projections = torch_projectors.project_2d_forw(
                reconstruction, rotations, shifts,
                output_shape=(128, 128),
                interpolation=interpolation,
                oversampling=2.0
            )
            loss = torch.sum(torch.abs(projections)**2)
            loss.backward()
            reconstruction.grad = None
            del projections, loss
        
        # Time both passes
        timing_results = self.time_forward_backward_operations(forward_pass, backward_pass)
        
        # Calculate throughput
        total_projections = self.num_reconstructions * self.num_projections_per_reconstruction
        forward_throughput = total_projections / timing_results["forward"]["median_time"] if timing_results["forward"]["median_time"] > 0 else 0.0
        total_throughput = total_projections / timing_results["forward_and_backward"]["median_time"] if timing_results["forward_and_backward"]["median_time"] > 0 else 0.0
        
        timing_results["throughput_forward_proj_per_sec"] = forward_throughput
        timing_results["throughput_total_proj_per_sec"] = total_throughput
        
        # Add memory profiling if enabled
        if self.profile_memory:
            memory_profile = self.profile_forward_backward_memory(forward_pass, backward_pass)
            timing_results["memory_profile"] = memory_profile
            
            # Add memory efficiency metrics
            if memory_profile:
                efficiency_metrics = self.calculate_memory_efficiency_metrics(memory_profile, total_projections)
                timing_results["memory_profile"].update(efficiency_metrics)
        
        return timing_results
    
    def benchmark_scenario_3(self, interpolation: str) -> dict:
        """
        Benchmark scenario 3: Real-space padding with oversampling=2.
        Start with 4096 different 128x65 tensors, irfft2d to real, pad to 256x256, rfft2d back, then oversampling=2.
        """
        print(f"      Running scenario 3 (real-space padding) with {interpolation} interpolation...")
        
        torch.manual_seed(42)
        
        # Generate 4096 different 128x65 complex tensors (RFFT format for 128x128 real)
        original_fourier = torch.randn(
            self.num_reconstructions, 128, 65, 
            dtype=torch.complex64, device=self.device
        )
        original_fourier.requires_grad_(True)
        
        # Convert to real space
        real_tensor = torch.fft.irfft2(original_fourier, s=(128, 128))
        
        # Pad to 256x256 (pad width: (left, right, top, bottom))
        padded_real = F.pad(real_tensor, (64, 64, 64, 64), mode='constant', value=0)
        
        # Convert back to Fourier space (256x129 format)
        reconstruction = torch.fft.rfft2(padded_real)
        
        rotations, shifts = self.generate_projection_poses()
        
        def forward_pass():
            """Forward projection operation with oversampling=2 after real-space padding."""
            # Convert to real space
            real_tensor = torch.fft.irfft2(original_fourier, s=(128, 128))
            
            # Pad to 256x256 (pad width: (left, right, top, bottom))
            padded_real = F.pad(real_tensor, (64, 64, 64, 64), mode='constant', value=0)
            
            # Convert back to Fourier space (256x129 format)
            reconstruction = torch.fft.rfft2(padded_real)

            projections = torch_projectors.project_2d_forw(
                reconstruction, rotations, shifts,
                output_shape=(128, 128),
                interpolation=interpolation,
                oversampling=2.0
            )
            del projections
            
        def backward_pass():
            """Combined forward + backward pass with oversampling=2 after real-space padding."""
            # Convert to real space
            real_tensor = torch.fft.irfft2(original_fourier, s=(128, 128))
            
            # Pad to 256x256 (pad width: (left, right, top, bottom))
            padded_real = F.pad(real_tensor, (64, 64, 64, 64), mode='constant', value=0)
            
            # Convert back to Fourier space (256x129 format)
            reconstruction = torch.fft.rfft2(padded_real)
            
            projections = torch_projectors.project_2d_forw(
                reconstruction, rotations, shifts,
                output_shape=(128, 128),
                interpolation=interpolation,
                oversampling=2.0
            )
            loss = torch.sum(torch.abs(projections)**2)
            loss.backward()
            reconstruction.grad = None
            del projections, loss
        
        # Time both passes
        timing_results = self.time_forward_backward_operations(forward_pass, backward_pass)
        
        # Calculate throughput
        total_projections = self.num_reconstructions * self.num_projections_per_reconstruction
        forward_throughput = total_projections / timing_results["forward"]["median_time"] if timing_results["forward"]["median_time"] > 0 else 0.0
        total_throughput = total_projections / timing_results["forward_and_backward"]["median_time"] if timing_results["forward_and_backward"]["median_time"] > 0 else 0.0
        
        timing_results["throughput_forward_proj_per_sec"] = forward_throughput
        timing_results["throughput_total_proj_per_sec"] = total_throughput
        
        # Add memory profiling if enabled
        if self.profile_memory:
            memory_profile = self.profile_forward_backward_memory(forward_pass, backward_pass)
            timing_results["memory_profile"] = memory_profile
            
            # Add memory efficiency metrics
            if memory_profile:
                efficiency_metrics = self.calculate_memory_efficiency_metrics(memory_profile, total_projections)
                timing_results["memory_profile"].update(efficiency_metrics)
        
        return timing_results
    
    def run_benchmarks(self, interpolations):
        """Run all benchmark scenarios with specified interpolations."""
        print(f"\nRunning oversampling benchmarks on {self.device}")
        print(f"Fixed parameters: num_reconstructions={self.num_reconstructions}, projections_per_reconstruction={self.num_projections_per_reconstruction}")
        print(f"Testing interpolations: {interpolations}")
        
        scenarios = [
            ("scenario_1", "128x128 basic", self.benchmark_scenario_1),
            ("scenario_2", "256x256 oversampling=2", self.benchmark_scenario_2), 
            ("scenario_3", "real-space padding + oversampling=2", self.benchmark_scenario_3)
        ]
        
        test_count = 0
        total_tests = len(scenarios) * len(interpolations)
        
        for scenario_id, scenario_desc, scenario_func in scenarios:
            for interpolation in interpolations:
                test_count += 1
                print(f"\n[{test_count}/{total_tests}] Testing {scenario_desc} with {interpolation} interpolation")
                
                try:
                    results = scenario_func(interpolation)
                    
                    print(f"    Forward: {results['forward']['median_time']:.4f}s")
                    print(f"    Backward: {results['backward']['median_time']:.4f}s") 
                    print(f"    Forward+Backward: {results['forward_and_backward']['median_time']:.4f}s")
                    print(f"    Throughput: {results['throughput_total_proj_per_sec']:.1f} proj/s")
                    
                    # Save results
                    test_name = f"{scenario_id}_{interpolation}"
                    self.add_benchmark_result(
                        test_name,
                        {
                            "scenario": scenario_id,
                            "scenario_description": scenario_desc,
                            "interpolation": interpolation,
                            "num_reconstructions": self.num_reconstructions,
                            "projections_per_reconstruction": self.num_projections_per_reconstruction
                        },
                        [],  # No raw times for nested results
                        {},  # No single stats
                        results
                    )
                    
                except Exception as e:
                    print(f"    ERROR: {e}")
                    # Save error result
                    error_results = {
                        "forward": {"median_time": None, "mean_time": None, "std_dev": None, "min_time": None, "max_time": None},
                        "backward": {"median_time": None, "mean_time": None, "std_dev": None, "min_time": None, "max_time": None},
                        "forward_and_backward": {"median_time": None, "mean_time": None, "std_dev": None, "min_time": None, "max_time": None},
                        "throughput_forward_proj_per_sec": None,
                        "throughput_total_proj_per_sec": None,
                        "error": str(e)
                    }
                    
                    test_name = f"{scenario_id}_{interpolation}"
                    self.add_benchmark_result(
                        test_name,
                        {
                            "scenario": scenario_id,
                            "scenario_description": scenario_desc,
                            "interpolation": interpolation,
                            "num_reconstructions": self.num_reconstructions,
                            "projections_per_reconstruction": self.num_projections_per_reconstruction
                        },
                        [],  # No raw times for nested results
                        {},  # No single stats
                        error_results
                    )
                
                # Cooldown between tests
                if self.test_cooldown_seconds > 0:
                    print(f"    Cooling down for {self.test_cooldown_seconds}s...")
                    time.sleep(self.test_cooldown_seconds)


def run_single_element_benchmark(benchmark: OversamplingBenchmark, args):
    """Run a single benchmark element for subprocess isolation."""
    # Extract interpolation (for subprocess mode)
    interpolation = getattr(args, 'interpolations', ['linear'])[0]
    
    print(f"Running single element with {interpolation} interpolation")
    
    # Run all three scenarios
    try:
        scenarios = [
            ("scenario_1", "128x128 basic", benchmark.benchmark_scenario_1),
            ("scenario_2", "256x256 oversampling=2", benchmark.benchmark_scenario_2),
            ("scenario_3", "real-space padding + oversampling=2", benchmark.benchmark_scenario_3)
        ]
        
        for scenario_id, scenario_desc, scenario_func in scenarios:
            results = scenario_func(interpolation)
            
            test_name = f"{scenario_id}_{interpolation}"
            benchmark.add_benchmark_result(
                test_name,
                {
                    "scenario": scenario_id,
                    "scenario_description": scenario_desc,
                    "interpolation": interpolation,
                    "num_reconstructions": benchmark.num_reconstructions,
                    "projections_per_reconstruction": benchmark.num_projections_per_reconstruction
                },
                [],  # No raw times for nested results
                {},  # No single stats
                results
            )
        
        print(f"Single element completed successfully")
        
    except Exception as e:
        print(f"Single element failed: {e}")
        raise


def main():
    parser = create_argument_parser("Benchmark oversampling scenarios in 2D forward projection")
    args = parser.parse_args()
    
    # Check if running in single-element mode (for subprocess isolation)
    if hasattr(args, 'single_element') and args.single_element:
        OversamplingBenchmark.run_single_element_mode(args, run_single_element_benchmark)
        return
    
    # Parse device
    device = BenchmarkBase.parse_device(args.device)
    
    # Create benchmark instance
    benchmark = OversamplingBenchmark(args.platform_name, device, args.title)
    
    print(f"Starting oversampling benchmark")
    print(f"Platform: {args.platform_name}")
    print(f"Device: {device}")
    print(f"Interpolations: {getattr(args, 'interpolations', ['linear', 'cubic'])}")
    print(f"Warmup runs: {args.warmup_runs}")
    print(f"Timing runs: {args.timing_runs}")
    print(f"Cooldown: {args.cooldown}s")
    print(f"Test cooldown: {args.test_cooldown}s")
    print(f"Memory profiling: {args.profile_memory}")
    
    # Override benchmark parameters if provided
    benchmark.warmup_runs = args.warmup_runs
    benchmark.timing_runs = args.timing_runs
    benchmark.cooldown_seconds = args.cooldown
    benchmark.test_cooldown_seconds = args.test_cooldown
    benchmark.profile_memory = args.profile_memory
    
    # Get interpolations to test
    interpolations = getattr(args, 'interpolations', ['linear', 'cubic'])
    
    # Run benchmarks
    try:
        benchmark.run_benchmarks(interpolations)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        raise
    finally:
        # Always save results
        benchmark.save_results()
        benchmark.print_summary()

    print(f"\nBenchmark completed successfully!")
    print(f"Results saved to: {benchmark.results_file}")


if __name__ == "__main__":
    main()