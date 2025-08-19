#!/usr/bin/env python3
"""
Benchmark script for 2D->3D back-projection performance.

This script tests torch_projectors.backproject_2d_to_3d_forw performance across different
configurations including batch sizes, image sizes, and interpolation methods.
Tests both gradient and no-gradient modes.

Usage:
    python backward_2d_to_3d.py --platform-name "a100-cuda" --device auto
    python backward_2d_to_3d.py --platform-name "m2-mps" --device mps --batch-sizes 1 4 --image-sizes 32 64
"""

import sys
import time
import torch
import torch_projectors
from pathlib import Path
from typing import Optional

# Add benchmark_base to path  
sys.path.append(str(Path(__file__).parent))
from benchmark_base import BenchmarkBase, create_argument_parser


class Backward2DTo3DBenchmark(BenchmarkBase):
    """Benchmark class for 2D->3D back-projection performance."""
    
    def __init__(self, platform_name: str, device: Optional[torch.device] = None, title: Optional[str] = None):
        super().__init__(platform_name, "backward_2d_to_3d", device, title)
        
    def generate_backproject_2d_to_3d_test_data(self, batch_size: int, image_size: int, 
                                               num_projections: int, depth: Optional[int] = None) -> tuple:
        """Generate consistent test data for 2D->3D back-projection benchmarks."""
        torch.manual_seed(42)  # Fixed seed for reproducibility
        
        if depth is None:
            depth = image_size  # Default to cubic volumes
        
        # 2D->3D back-projection takes 2D projections as input
        # Projections in RFFT format: [batch, num_projections, height, width//2 + 1]
        projections = torch.randn(
            batch_size, num_projections, image_size, image_size // 2 + 1,
            dtype=torch.complex64, device='cpu'
        ).to(self.device)
        
        # Optional weights for weighted back-projection
        weights = torch.rand(
            batch_size, num_projections, image_size, image_size // 2 + 1,
            dtype=torch.float32, device='cpu'
        ).to(self.device)
        
        # 3D rotation matrices [batch, num_projections, 3, 3]
        rotations = self._generate_3d_rotations(batch_size, num_projections)
        
        # Random shifts [batch, num_projections, 2]
        shifts = torch.randn(batch_size, num_projections, 2, device='cpu').to(self.device) * 5.0
        
        return projections, weights, rotations, shifts, depth
        
    def benchmark_with_gradients(self, batch_size: int, image_size: int, 
                                num_projections: int, interpolation: str) -> dict:
        """Benchmark 2D->3D back-projection with gradient computation enabled."""
        # Generate test data for 2D->3D back-projection
        projections, weights, rotations, shifts, depth = self.generate_backproject_2d_to_3d_test_data(
            batch_size, image_size, num_projections
        )
        
        def forward_pass():
            """Forward 2D->3D back-projection operation."""
            projections.requires_grad_(True)
            weights.requires_grad_(True)
            reconstruction, weight_reconstruction = torch_projectors.backproject_2d_to_3d_forw(
                projections, rotations, weights=weights, shifts=shifts,
                interpolation=interpolation
            )
            # Don't return tensors to avoid memory profiler issues
            del reconstruction, weight_reconstruction
            
        def backward_pass():
            """Combined forward + backward pass."""
            projections.requires_grad_(True)
            weights.requires_grad_(True)
            reconstruction, weight_reconstruction = torch_projectors.backproject_2d_to_3d_forw(
                projections, rotations, weights=weights, shifts=shifts,
                interpolation=interpolation
            )
            # Loss includes both outputs
            loss = torch.sum(torch.abs(reconstruction)**2) + 0.1 * torch.sum(weight_reconstruction**2)
            loss.backward()
            projections.grad = None
            weights.grad = None
            del reconstruction, weight_reconstruction, loss
            
        # Reset memory stats
        self.reset_memory_stats()
        
        # Time forward pass
        forward_times, forward_stats = self.time_function(forward_pass)
        
        # Time backward pass (forward + backward)
        backward_times, backward_stats = self.time_function(backward_pass)
        
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
        
        # Calculate throughput (total back-projections processed)
        total_backprojections = batch_size * num_projections
        forward_throughput = total_backprojections / forward_stats["median_time"] if forward_stats["median_time"] > 0 else 0.0
        total_throughput = total_backprojections / backward_stats["median_time"] if backward_stats["median_time"] > 0 else 0.0
        
        results = {
            "forward": forward_stats,
            "backward": pure_backward_stats,
            "forward_and_backward": backward_stats,
            "throughput_forward_backproj_per_sec": forward_throughput,
            "throughput_total_backproj_per_sec": total_throughput,
            **peak_memory
        }
        
        # Add memory profiling if enabled  
        if self.profile_memory:
            print(f"      Profiling memory usage...")
            
            # Profile forward pass with detailed memory tracking
            forward_memory = self.profile_memory_usage_detailed(forward_pass)
            
            # Profile backward pass with detailed memory tracking  
            backward_memory = self.profile_memory_usage_detailed(backward_pass)
            
            # Calculate actual tensor sizes with gradient information
            input_sizes = self.calculate_tensor_sizes(projections, weights, rotations, shifts, include_gradients=True)
            
            # Create sample outputs to measure size
            with torch.no_grad():
                sample_reconstruction, sample_weight_reconstruction = torch_projectors.backproject_2d_to_3d_forw(
                    projections[:1, :1], rotations[:1, :1], weights=weights[:1, :1], 
                    shifts=shifts[:1, :1], interpolation=interpolation
                )
            output_sizes = self.calculate_tensor_sizes(sample_reconstruction, sample_weight_reconstruction, include_gradients=False)
            # Scale output size to match full batch (3D reconstructions scale with batch_size but not num_projections)
            output_sizes["total_mb"] = output_sizes["total_mb"] * batch_size
            
            # Also measure the expected gradient sizes after backward pass
            test_proj = projections[:1, :1].clone().requires_grad_(True)
            test_weights = weights[:1, :1].clone().requires_grad_(True)
            test_recon, test_weight_recon = torch_projectors.backproject_2d_to_3d_forw(
                test_proj, rotations[:1, :1], weights=test_weights, 
                shifts=shifts[:1, :1], interpolation=interpolation
            )
            test_loss = torch.sum(torch.abs(test_recon)**2) + 0.1 * torch.sum(test_weight_recon**2)
            test_loss.backward()
            
            # Measure gradient sizes
            gradient_sizes = self.calculate_tensor_sizes(test_proj, test_weights, include_gradients=True)
            
            # Clean up test tensors
            del test_proj, test_weights, test_recon, test_weight_recon, test_loss
            
            results["memory_profile"] = {
                "forward_memory": forward_memory,
                "backward_memory": backward_memory,
                "input_data_sizes": input_sizes,
                "output_data_sizes": output_sizes,
                "gradient_data_sizes": gradient_sizes
            }
            
            # Memory efficiency metrics
            if "peak_gpu_memory_mb" in forward_memory:
                results["memory_profile"]["forward_memory_per_backproj_mb"] = forward_memory["peak_gpu_memory_mb"] / total_backprojections
                results["memory_profile"]["forward_memory_efficiency"] = forward_memory["peak_gpu_memory_mb"] / input_sizes["total_mb"]
                results["memory_profile"]["io_ratio"] = output_sizes["total_mb"] / input_sizes["total_mb"] if input_sizes["total_mb"] > 0 else 0.0
            
        return results
    
    def benchmark_without_gradients(self, batch_size: int, image_size: int,
                                  num_projections: int, interpolation: str) -> dict:
        """Benchmark 2D->3D back-projection without gradients (inference mode)."""
        # Generate test data for 2D->3D back-projection
        projections, weights, rotations, shifts, depth = self.generate_backproject_2d_to_3d_test_data(
            batch_size, image_size, num_projections
        )
        
        def forward_only():
            """Forward-only 2D->3D back-projection operation without gradients."""
            with torch.no_grad():
                reconstruction, weight_reconstruction = torch_projectors.backproject_2d_to_3d_forw(
                    projections, rotations, weights=weights, shifts=shifts,
                    interpolation=interpolation
                )
                del reconstruction, weight_reconstruction
        
        # Reset memory stats
        self.reset_memory_stats()
        
        # Time forward pass
        _, forward_stats = self.time_function(forward_only)
        
        # Get peak memory usage
        peak_memory = self.get_peak_memory()
        
        # Calculate throughput
        total_backprojections = batch_size * num_projections
        throughput = total_backprojections / forward_stats["median_time"] if forward_stats["median_time"] > 0 else 0.0
        
        results = {
            "forward_no_grad": forward_stats,
            "throughput_backproj_per_sec": throughput,
            **peak_memory
        }
        
        # Add memory profiling if enabled
        if self.profile_memory:
            print(f"      Profiling memory usage (no-grad)...")
            
            # Profile forward pass (no gradients) with detailed memory tracking
            forward_memory = self.profile_memory_usage_detailed(forward_only)
            
            # Calculate actual tensor sizes
            input_sizes = self.calculate_tensor_sizes(projections, weights, rotations, shifts)
            # Use sample back-projection for size calculation
            with torch.no_grad():
                sample_reconstruction, sample_weight_reconstruction = torch_projectors.backproject_2d_to_3d_forw(
                    projections[:1, :1], rotations[:1, :1], weights=weights[:1, :1],
                    shifts=shifts[:1, :1], interpolation=interpolation
                )
            output_sizes = self.calculate_tensor_sizes(sample_reconstruction, sample_weight_reconstruction)
            # Scale output size to match full batch (3D reconstructions scale with batch_size)
            output_sizes["total_mb"] = output_sizes["total_mb"] * batch_size
            
            results["memory_profile"] = {
                "forward_memory": forward_memory,
                "input_data_sizes": input_sizes,
                "output_data_sizes": output_sizes
            }
            
            # Memory efficiency metrics
            if "peak_gpu_memory_mb" in forward_memory:
                results["memory_profile"]["forward_memory_per_backproj_mb"] = forward_memory["peak_gpu_memory_mb"] / total_backprojections
                results["memory_profile"]["forward_memory_efficiency"] = forward_memory["peak_gpu_memory_mb"] / input_sizes["total_mb"]
                results["memory_profile"]["io_ratio"] = output_sizes["total_mb"] / input_sizes["total_mb"] if input_sizes["total_mb"] > 0 else 0.0
            
        return results
    
    def run_benchmarks(self, batch_sizes, image_sizes, num_projections_list, interpolations):
        """Run all benchmark combinations with thermal-aware ordering."""
        print(f"\nRunning 2D->3D back-projection benchmarks on {self.device}")
        print(f"Test matrix: {len(batch_sizes)} batch sizes × {len(image_sizes)} image sizes × {len(num_projections_list)} projection counts × {len(interpolations)} interpolations")
        print(f"Total tests: {len(batch_sizes) * len(image_sizes) * len(num_projections_list) * len(interpolations) * 2} (with/without gradients)")
        
        # Generate parameter matrix
        parameter_matrix = []
        for batch_size in batch_sizes:
            for image_size in image_sizes:
                for num_projections in num_projections_list:
                    for interpolation in interpolations:
                        parameter_matrix.append({
                            "batch_size": batch_size,
                            "image_size": image_size,
                            "num_projections": num_projections,
                            "interpolation": interpolation
                        })
        
        # Apply thermal-aware zigzag ordering
        print(f"Applying thermal-aware ordering (alternating light/heavy tests)...")
        ordered_params = BenchmarkBase.create_zigzag_ordering(parameter_matrix)
        
        # Show thermal load distribution
        loads = [BenchmarkBase.calculate_thermal_load(p["batch_size"], p["image_size"], p["num_projections"], p["interpolation"]) 
                for p in ordered_params]
        print(f"Thermal load range: {min(loads):.0f} to {max(loads):.0f}")
        print(f"First few test loads: {[f'{load:.0f}' for load in loads[:6]]}")
        
        test_count = 0
        total_tests = len(ordered_params) * 2
        
        for params in ordered_params:
            batch_size = params["batch_size"]
            image_size = params["image_size"] 
            num_projections = params["num_projections"]
            interpolation = params["interpolation"]
            
            thermal_load = BenchmarkBase.calculate_thermal_load(batch_size, image_size, num_projections, interpolation)
            print(f"\nTesting: batch={batch_size}, size={image_size}x{image_size}, projs={num_projections}, interp={interpolation} (load={thermal_load:.0f})")
            
            # Test with gradients
            test_count += 1
            print(f"  [{test_count}/{total_tests}] With gradients...")
            
            try:
                with_grad_results = self.benchmark_with_gradients(
                    batch_size, image_size, num_projections, interpolation
                )
                
                print(f"    Forward: {with_grad_results['forward']['median_time']:.4f}s")
                print(f"    Backward: {with_grad_results['backward']['median_time']:.4f}s")
                print(f"    Forward+Backward: {with_grad_results['forward_and_backward']['median_time']:.4f}s")
                print(f"    Throughput: {with_grad_results['throughput_total_backproj_per_sec']:.1f} backproj/s")
                
            except Exception as e:
                print(f"    ERROR in with-gradients test: {e}")
                # Create failed result structure with None values
                with_grad_results = {
                    "forward": {"median_time": None, "mean_time": None, "std_dev": None, "min_time": None, "max_time": None},
                    "backward": {"median_time": None, "mean_time": None, "std_dev": None, "min_time": None, "max_time": None},
                    "forward_and_backward": {"median_time": None, "mean_time": None, "std_dev": None, "min_time": None, "max_time": None},
                    "throughput_forward_backproj_per_sec": None,
                    "throughput_total_backproj_per_sec": None,
                    "error": str(e)
                }
            
            # Save with-gradient results (successful or failed)
            test_name = f"backward_2d_to_3d_{interpolation}_{image_size}x{image_size}x{image_size}_batch{batch_size}_proj{num_projections}_grad"
            self.add_benchmark_result(
                test_name,
                {
                    "operation": "backward_2d_to_3d",
                    "batch_size": batch_size,
                    "image_size": [image_size, image_size],
                    "volume_size": [image_size, image_size, image_size],
                    "num_projections": num_projections,
                    "interpolation": interpolation,
                    "with_gradients": True
                },
                [],  # No raw times for nested results
                {},  # No single stats
                with_grad_results
            )
            
            # Test without gradients
            test_count += 1
            print(f"  [{test_count}/{total_tests}] Without gradients...")
            
            try:
                no_grad_results = self.benchmark_without_gradients(
                    batch_size, image_size, num_projections, interpolation
                )
                
                print(f"    Forward (no-grad): {no_grad_results['forward_no_grad']['median_time']:.4f}s")
                print(f"    Throughput: {no_grad_results['throughput_backproj_per_sec']:.1f} backproj/s")
                
            except Exception as e:
                print(f"    ERROR in no-gradients test: {e}")
                # Create failed result structure with None values
                no_grad_results = {
                    "forward_no_grad": {"median_time": None, "mean_time": None, "std_dev": None, "min_time": None, "max_time": None},
                    "throughput_backproj_per_sec": None,
                    "error": str(e)
                }
            
            # Save no-gradient results (successful or failed)
            test_name = f"backward_2d_to_3d_{interpolation}_{image_size}x{image_size}x{image_size}_batch{batch_size}_proj{num_projections}_nograd"
            self.add_benchmark_result(
                test_name,
                {
                    "operation": "backward_2d_to_3d", 
                    "batch_size": batch_size,
                    "image_size": [image_size, image_size],
                    "volume_size": [image_size, image_size, image_size],
                    "num_projections": num_projections,
                    "interpolation": interpolation,
                    "with_gradients": False
                },
                [],  # No raw times for nested results
                {},  # No single stats
                no_grad_results
            )
            
            # Test matrix element cooldown to prevent thermal throttling
            if self.test_cooldown_seconds > 0:
                print(f"    Cooling down for {self.test_cooldown_seconds}s...")
                time.sleep(self.test_cooldown_seconds)


def run_single_element_benchmark(benchmark: Backward2DTo3DBenchmark, args):
    """Run a single benchmark element for subprocess isolation."""
    # Extract single values from lists (for subprocess mode)
    batch_size = args.batch_sizes[0] if args.batch_sizes else 1
    image_size = args.image_sizes[0] if args.image_sizes else 32
    num_projections = getattr(args, 'num_projections', [64])[0]
    interpolation = getattr(args, 'interpolations', ['linear'])[0]
    
    print(f"Running single element: batch={batch_size}, size={image_size}x{image_size}x{image_size}, projs={num_projections}, interp={interpolation}")
    
    # Run both with and without gradients tests
    try:
        # Test with gradients
        with_grad_results = benchmark.benchmark_with_gradients(
            batch_size, image_size, num_projections, interpolation
        )
        
        test_name_grad = f"backward_2d_to_3d_{interpolation}_{image_size}x{image_size}x{image_size}_batch{batch_size}_proj{num_projections}_grad"
        benchmark.add_benchmark_result(
            test_name_grad,
            {
                "operation": "backward_2d_to_3d",
                "batch_size": batch_size,
                "image_size": [image_size, image_size],
                "volume_size": [image_size, image_size, image_size],
                "num_projections": num_projections,
                "interpolation": interpolation,
                "with_gradients": True
            },
            [],  # No raw times for nested results
            {},  # No single stats
            with_grad_results
        )
        
        # Test without gradients
        no_grad_results = benchmark.benchmark_without_gradients(
            batch_size, image_size, num_projections, interpolation
        )
        
        test_name_nograd = f"backward_2d_to_3d_{interpolation}_{image_size}x{image_size}x{image_size}_batch{batch_size}_proj{num_projections}_nograd"
        benchmark.add_benchmark_result(
            test_name_nograd,
            {
                "operation": "backward_2d_to_3d", 
                "batch_size": batch_size,
                "image_size": [image_size, image_size],
                "volume_size": [image_size, image_size, image_size],
                "num_projections": num_projections,
                "interpolation": interpolation,
                "with_gradients": False
            },
            [],  # No raw times for nested results
            {},  # No single stats
            no_grad_results
        )
        
        print(f"Single element completed successfully")
        
    except Exception as e:
        print(f"Single element failed: {e}")
        raise


def main():
    parser = create_argument_parser("Benchmark 2D->3D back-projection performance")
    args = parser.parse_args()
    
    # Check if running in single-element mode (for subprocess isolation)
    if hasattr(args, 'single_element') and args.single_element:
        Backward2DTo3DBenchmark.run_single_element_mode(args, run_single_element_benchmark)
        return
    
    # Parse device
    device = BenchmarkBase.parse_device(args.device)
    
    # Create benchmark instance
    benchmark = Backward2DTo3DBenchmark(args.platform_name, device, args.title)
    
    print(f"Starting 2D->3D back-projection benchmarks")
    print(f"Platform: {args.platform_name}")
    print(f"Device: {device}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Image sizes: {args.image_sizes}")
    print(f"Num projections: {getattr(args, 'num_projections', [2, 64, 256, 1024])}")
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
    
    # Generate parameter matrix
    batch_sizes = args.batch_sizes
    image_sizes = args.image_sizes
    num_projections_list = getattr(args, 'num_projections', [2, 64, 256, 1024])
    interpolations = getattr(args, 'interpolations', ['linear', 'cubic'])
    
    parameter_matrix = []
    for batch_size in batch_sizes:
        for image_size in image_sizes:
            for num_projections in num_projections_list:
                for interpolation in interpolations:
                    parameter_matrix.append({
                        "batch_size": batch_size,
                        "image_size": image_size,
                        "num_projections": num_projections,
                        "interpolation": interpolation
                    })
    
    print(f"Generated parameter matrix with {len(parameter_matrix)} elements")
    
    # Apply thermal-aware zigzag ordering to both execution paths
    print(f"Applying thermal-aware ordering (alternating light/heavy tests)...")
    parameter_matrix = BenchmarkBase.create_zigzag_ordering(parameter_matrix)
    
    # Show thermal load distribution
    loads = [BenchmarkBase.calculate_thermal_load(p["batch_size"], p["image_size"], p["num_projections"], p["interpolation"]) 
            for p in parameter_matrix]
    print(f"Thermal load range: {min(loads):.0f} to {max(loads):.0f}")
    print(f"First few test loads: {[f'{load:.0f}' for load in loads[:6]]}")
    
    # Run benchmarks with subprocess isolation
    try:
        script_path = str(Path(__file__).resolve())
        if benchmark.use_subprocess_isolation:
            print("Using subprocess isolation for memory management")
            benchmark.results = benchmark.run_matrix_with_subprocess_isolation(
                script_path, parameter_matrix
            )
        else:
            print("Using traditional in-process execution")
            benchmark.run_benchmarks(
                batch_sizes, image_sizes, num_projections_list, interpolations
            )
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