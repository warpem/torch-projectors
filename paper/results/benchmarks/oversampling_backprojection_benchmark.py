#!/usr/bin/env python3
"""
Benchmark script for oversampling scenarios in 2D backward projection.

This script tests torch_projectors.backproject_2d_forw performance with different
oversampling configurations and padding strategies:
1. Basic 128x128 backward projection with 4096 projections
2. 256x256 with oversampling=2 and 128x128 output
3. Real-space cropping: backproject with oversampling=2, then crop reconstruction

Usage:
    python oversampling_backprojection_benchmark.py --platform-name "a100-cuda" --device auto
    python oversampling_backprojection_benchmark.py --platform-name "m2-mps" --device mps --interpolations linear cubic
"""

import sys
import time
import math
import torch
import torch_projectors
from pathlib import Path
from typing import Optional

# Add benchmark_base to path  
sys.path.append(str(Path(__file__).parent))
from benchmark_base import BenchmarkBase, create_argument_parser


class OversamplingBackprojectionBenchmark(BenchmarkBase):
    """Benchmark class for backprojection oversampling scenarios."""
    
    def __init__(self, platform_name: str, device: Optional[torch.device] = None, title: Optional[str] = None):
        super().__init__(platform_name, "oversampling_backprojection_benchmark", device, title)
        
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
        Benchmark scenario 1: Basic 128x128 backward projection.
        Start with 16384 different 128x65 complex projection tensors (RFFT format), 1 projection each.
        """
        print(f"      Running scenario 1 (128x128 basic) with {interpolation} interpolation...")
        
        torch.manual_seed(42)
        
        # Generate 16384 different 128x65 complex tensors (RFFT format for 128x128 real projections)
        projections = torch.randn(
            self.num_reconstructions, self.num_projections_per_reconstruction, 128, 65, 
            dtype=torch.complex64, device=self.device
        )
        
        rotations, shifts = self.generate_projection_poses()
        
        def forward_pass():
            """Forward backprojection operation."""
            projections.requires_grad_(True)
            data_rec, weight_rec = torch_projectors.backproject_2d_forw(
                projections, rotations, shifts=shifts,
                interpolation=interpolation
            )
            del data_rec, weight_rec
            
        def backward_pass():
            """Combined forward + backward pass."""
            projections.requires_grad_(True)
            data_rec, weight_rec = torch_projectors.backproject_2d_forw(
                projections, rotations, shifts=shifts,
                interpolation=interpolation
            )
            loss = torch.sum(torch.abs(data_rec)**2)
            loss.backward()
            projections.grad = None
            del data_rec, weight_rec, loss
        
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
        Start with 16384 different 128x65 complex projection tensors, use oversampling=2, output 256x129.
        """
        print(f"      Running scenario 2 (256x256 oversampling=2) with {interpolation} interpolation...")
        
        torch.manual_seed(42)
        
        # Generate 16384 different 128x65 complex tensors (RFFT format for 128x128 real projections)
        projections = torch.randn(
            self.num_reconstructions, self.num_projections_per_reconstruction, 128, 65, 
            dtype=torch.complex64, device=self.device
        )
        
        rotations, shifts = self.generate_projection_poses()
        
        def forward_pass():
            """Forward backprojection operation with oversampling=2."""
            projections.requires_grad_(True)
            data_rec, weight_rec = torch_projectors.backproject_2d_forw(
                projections, rotations, shifts=shifts,
                interpolation=interpolation,
                oversampling=2.0
            )
            del data_rec, weight_rec
            
        def backward_pass():
            """Combined forward + backward pass with oversampling=2."""
            projections.requires_grad_(True)
            data_rec, weight_rec = torch_projectors.backproject_2d_forw(
                projections, rotations, shifts=shifts,
                interpolation=interpolation,
                oversampling=2.0
            )
            loss = torch.sum(torch.abs(data_rec)**2)
            loss.backward()
            projections.grad = None
            del data_rec, weight_rec, loss
        
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
        Benchmark scenario 3: Real-space cropping after backprojection.
        Start with 16384 different 128x65 projection tensors, backproject with oversampling=2, 
        then take reconstruction, irfft2, crop to 128x128, rfft2 back.
        """
        print(f"      Running scenario 3 (real-space cropping) with {interpolation} interpolation...")
        
        torch.manual_seed(42)
        
        # Generate 16384 different 128x65 complex tensors (RFFT format for 128x128 real projections)
        projections = torch.randn(
            self.num_reconstructions, self.num_projections_per_reconstruction, 128, 65, 
            dtype=torch.complex64, device=self.device
        )
        projections.requires_grad_(True)
        
        rotations, shifts = self.generate_projection_poses()
        
        def forward_pass():
            """Forward backprojection operation with oversampling=2 followed by real-space cropping."""
            # Backproject with oversampling=2
            data_rec, weight_rec = torch_projectors.backproject_2d_forw(
                projections, rotations, shifts=shifts,
                interpolation=interpolation,
                oversampling=2.0
            )
            
            # Convert to real space (256x256)
            real_tensor = torch.fft.irfft2(data_rec, s=(256, 256))
            
            # Crop to 128x128 (remove 64 pixels from each side)
            cropped_real = real_tensor[:, 64:192, 64:192]
            
            # Convert back to Fourier space (128x65 format)
            reconstruction = torch.fft.rfft2(cropped_real)
            
            del data_rec, weight_rec, real_tensor, cropped_real, reconstruction
            
        def backward_pass():
            """Combined forward + backward pass with oversampling=2 followed by real-space cropping."""
            # Backproject with oversampling=2
            data_rec, weight_rec = torch_projectors.backproject_2d_forw(
                projections, rotations, shifts=shifts,
                interpolation=interpolation,
                oversampling=2.0
            )
            
            # Convert to real space (256x256)
            real_tensor = torch.fft.irfft2(data_rec, s=(256, 256))
            
            # Crop to 128x128 (remove 64 pixels from each side)
            cropped_real = real_tensor[:, 64:192, 64:192]
            
            # Convert back to Fourier space (128x65 format)
            reconstruction = torch.fft.rfft2(cropped_real)
            
            loss = torch.sum(torch.abs(reconstruction)**2)
            loss.backward()
            projections.grad = None
            del data_rec, weight_rec, real_tensor, cropped_real, reconstruction, loss
        
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
        print(f"\nRunning oversampling backprojection benchmarks on {self.device}")
        print(f"Fixed parameters: num_reconstructions={self.num_reconstructions}, projections_per_reconstruction={self.num_projections_per_reconstruction}")
        print(f"Testing interpolations: {interpolations}")
        
        scenarios = [
            ("scenario_1", "128x128 basic backprojection", self.benchmark_scenario_1),
            ("scenario_2", "256x256 oversampling=2", self.benchmark_scenario_2), 
            ("scenario_3", "real-space cropping + oversampling=2", self.benchmark_scenario_3)
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


def run_single_element_benchmark(benchmark: OversamplingBackprojectionBenchmark, args):
    """Run a single benchmark element for subprocess isolation."""
    # Extract interpolation (for subprocess mode)
    interpolation = getattr(args, 'interpolations', ['linear'])[0]
    
    print(f"Running single element with {interpolation} interpolation")
    
    # Run all three scenarios
    try:
        scenarios = [
            ("scenario_1", "128x128 basic backprojection", benchmark.benchmark_scenario_1),
            ("scenario_2", "256x256 oversampling=2", benchmark.benchmark_scenario_2),
            ("scenario_3", "real-space cropping + oversampling=2", benchmark.benchmark_scenario_3)
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
    parser = create_argument_parser("Benchmark oversampling scenarios in 2D backward projection")
    args = parser.parse_args()
    
    # Check if running in single-element mode (for subprocess isolation)
    if hasattr(args, 'single_element') and args.single_element:
        OversamplingBackprojectionBenchmark.run_single_element_mode(args, run_single_element_benchmark)
        return
    
    # Parse device
    device = BenchmarkBase.parse_device(args.device)
    
    # Create benchmark instance
    benchmark = OversamplingBackprojectionBenchmark(args.platform_name, device, args.title)
    
    print(f"Starting oversampling backprojection benchmark")
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