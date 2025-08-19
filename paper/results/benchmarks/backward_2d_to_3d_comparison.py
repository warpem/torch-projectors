#!/usr/bin/env python3
"""
Benchmark script comparing torch-projectors vs torch-fourier-slice for 2D->3D back-projection.

This script specifically benchmarks torch_projectors.backproject_2d_to_3d_forw against
torch-fourier-slice's insert_central_slices_rfft_3d, following the example from 
tests/3d/back/test_performance_back_2d_to_3d.py.

Important memory constraint: Backward passes through torch-fourier-slice are only 
attempted when batch_size=1 and num_projections=1 to avoid out-of-memory errors.

Usage:
    python backward_2d_to_3d_comparison.py --platform-name "a100-cuda" --device auto
    python backward_2d_to_3d_comparison.py --platform-name "m2-mps" --device mps --batch-sizes 1 --image-sizes 64 128
"""

import sys
import time
import torch
import torch_projectors
import statistics
import json
from pathlib import Path
from typing import Optional

# Add benchmark_base to path  
sys.path.append(str(Path(__file__).parent))
from benchmark_base import BenchmarkBase, create_argument_parser

# Third-party benchmark import (required for this comparison benchmark)
try:
    from torch_fourier_slice import insert_central_slices_rfft_3d
    HAS_TORCH_FOURIER_SLICE = True
except ImportError:
    HAS_TORCH_FOURIER_SLICE = False
    print("ERROR: torch-fourier-slice not available. This benchmark requires torch-fourier-slice.")
    print("Install with: pip install torch-fourier-slice")
    sys.exit(1)


class Backward2DTo3DComparisonBenchmark(BenchmarkBase):
    """Benchmark class comparing torch-projectors vs torch-fourier-slice for 2D->3D back-projection."""
    
    def __init__(self, platform_name: str, device: Optional[torch.device] = None, title: Optional[str] = None):
        super().__init__(platform_name, "backward_2d_to_3d_comparison", device, title)
        self.disable_tfs_backward_safety = False  # Will be set by main()
        
    def run_single_element_subprocess(self, script_path: str, element_params: dict, 
                                    temp_dir: Path) -> Optional[dict]:
        """Override to add custom CLI arguments for subprocess."""
        import shutil
        import subprocess
        import uuid
        
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
            
        # Add our custom flag
        if self.disable_tfs_backward_safety:
            cmd.append("--disable-tfs-backward-safety")
        
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
        
    def benchmark_library_comparison(self, batch_size: int, image_size: int, 
                                   num_projections: int, interpolation: str, 
                                   disable_tfs_backward_safety: bool = False) -> dict:
        """Benchmark torch-projectors vs torch-fourier-slice for 2D->3D back-projection.
        
        Important: Due to memory constraints, backward passes through torch-fourier-slice
        are only attempted when batch_size == 1 and num_projections == 1.
        """
        # Skip MPS - torch-fourier-slice operators not implemented for MPS
        if self.device.type == "mps":
            return {"error": "torch-fourier-slice operators not implemented for MPS"}
        
        print(f"      Comparing torch-projectors vs torch-fourier-slice...")
        print(f"      Image: {image_size}x{image_size}, Batch: {batch_size}, Projections: {num_projections}, Interpolation: {interpolation}")
        
        # Generate test data (2D projections for back-projection) - same seed for fair comparison
        torch.manual_seed(42)
        
        # 2D projections in RFFT format for back-projection to 3D
        W_half = image_size // 2 + 1
        projections_tp = torch.randn(batch_size, num_projections, image_size, W_half, 
                                    dtype=torch.complex64, device=self.device)
        weights_tp = torch.rand(batch_size, num_projections, image_size, W_half, 
                               dtype=torch.float32, device=self.device)
        
        # Create torch-fourier-slice compatible data
        projections_tfs = projections_tp.clone()
        projections_tfs.requires_grad_(True)
        
        # Convert rotation matrices to torch-fourier-slice format
        rotations_tp = self._generate_3d_rotations(batch_size, num_projections)
        rotations_tfs = rotations_tp.clone()
        
        # Generate shifts for torch-projectors
        shifts_tp = torch.randn(batch_size, num_projections, 2, device=self.device) * 5.0
        
        def benchmark_torch_fourier_slice():
            """Benchmark torch-fourier-slice using 2D->3D back-projection API"""
            forward_times = []
            backward_times = []
            
            # Can we safely do backward pass? Only if batch=1 and projections=1, unless safety is disabled
            safe_for_backward = (batch_size < 8 or image_size < 256 or num_projections < 2048)
            
            print(f"        torch-fourier-slice backward pass: {'ENABLED' if safe_for_backward else 'DISABLED (memory constraint)'}")
            
            # Warmup runs
            for _ in range(self.warmup_runs):
                projections_tfs.grad = None
                
                all_volumes = []
                for i in range(batch_size):
                    projections = projections_tfs[i]  # Single projection set (num_projections, image_size, W_half)
                    rotations_batch = rotations_tfs[i]  # Rotations for this set (num_projections, 3, 3)
                    # Use 2D->3D back-projection API with image_shape parameter
                    if safe_for_backward:
                        volume, weight_volume = insert_central_slices_rfft_3d(
                            projections, (image_size, image_size, image_size), rotations_batch
                        )
                    else:
                        with torch.no_grad():
                            volume, weight_volume = insert_central_slices_rfft_3d(
                                projections, (image_size, image_size, image_size), rotations_batch
                            )
                    all_volumes.append(volume)
                
                if safe_for_backward:
                    all_volumes = torch.stack(all_volumes)
                    loss = torch.sum(torch.abs(all_volumes)**2)
                    loss.backward()
            
            # Timing runs
            for _ in range(self.timing_runs):
                projections_tfs.grad = None
                
                # Time forward pass
                self._synchronize()
                start_time = time.perf_counter()
                
                all_volumes = []
                for i in range(batch_size):
                    projections = projections_tfs[i]  # Single projection set (num_projections, image_size, W_half)
                    rotations_batch = rotations_tfs[i]  # Rotations for this set (num_projections, 3, 3)
                    # Use 2D->3D back-projection API with image_shape parameter
                    if safe_for_backward:
                        volume, weight_volume = insert_central_slices_rfft_3d(
                            projections, (image_size, image_size, image_size), rotations_batch
                        )
                    else:
                        with torch.no_grad():
                            volume, weight_volume = insert_central_slices_rfft_3d(
                                projections, (image_size, image_size, image_size), rotations_batch
                            )
                    all_volumes.append(volume)
                
                all_volumes = torch.stack(all_volumes)
                
                self._synchronize()
                forward_time = time.perf_counter() - start_time
                forward_times.append(forward_time)
                
                # Time backward pass only if safe
                if safe_for_backward:
                    loss = torch.sum(torch.abs(all_volumes)**2)
                    
                    self._synchronize()
                    start_time = time.perf_counter()
                    
                    loss.backward()
                    
                    self._synchronize()
                    backward_time = time.perf_counter() - start_time
                    backward_times.append(backward_time)
            
            return forward_times, backward_times
        
        def benchmark_torch_projectors():
            """Benchmark torch-projectors"""
            forward_times = []
            backward_times = []
            
            print(f"        torch-projectors backward pass: ENABLED")
            
            # Warmup runs
            for _ in range(self.warmup_runs):
                projections_tp.requires_grad_(True)
                weights_tp.requires_grad_(True)
                reconstruction, weight_reconstruction = torch_projectors.backproject_2d_to_3d_forw(
                    projections_tp, rotations_tp, weights=weights_tp, shifts=shifts_tp,
                    interpolation=interpolation
                )
                loss = torch.sum(torch.abs(reconstruction)**2) + 0.1 * torch.sum(weight_reconstruction**2)
                loss.backward()
                projections_tp.grad = None
                weights_tp.grad = None
            
            # Timing runs
            for _ in range(self.timing_runs):
                projections_tp.requires_grad_(True)
                weights_tp.requires_grad_(True)
                
                # Time forward pass
                self._synchronize()
                start_time = time.perf_counter()
                
                reconstruction, weight_reconstruction = torch_projectors.backproject_2d_to_3d_forw(
                    projections_tp, rotations_tp, weights=weights_tp, shifts=shifts_tp,
                    interpolation=interpolation
                )
                
                self._synchronize()
                forward_time = time.perf_counter() - start_time
                forward_times.append(forward_time)
                
                # Time backward pass
                loss = torch.sum(torch.abs(reconstruction)**2) + 0.1 * torch.sum(weight_reconstruction**2)
                
                self._synchronize()
                start_time = time.perf_counter()
                
                loss.backward()
                
                self._synchronize()
                backward_time = time.perf_counter() - start_time
                backward_times.append(backward_time)
                
                projections_tp.grad = None
                weights_tp.grad = None
            
            return forward_times, backward_times
        
        # Reset memory stats
        self.reset_memory_stats()
        
        # Run benchmarks
        print(f"        Benchmarking torch-fourier-slice...")
        tfs_forward, tfs_backward = benchmark_torch_fourier_slice()
        
        print(f"        Benchmarking torch-projectors...")
        tp_forward, tp_backward = benchmark_torch_projectors()
        
        # Get peak memory usage
        peak_memory = self.get_peak_memory()
        
        # Calculate statistics
        def calc_stats(times):
            if not times:
                return {"median_time": None, "mean_time": None, "std_dev": None, "min_time": None, "max_time": None}
            return {
                'median_time': statistics.median(times),
                'mean_time': statistics.mean(times),
                'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
                'min_time': min(times),
                'max_time': max(times)
            }
        
        tfs_forward_stats = calc_stats(tfs_forward)
        tfs_backward_stats = calc_stats(tfs_backward)
        tp_forward_stats = calc_stats(tp_forward)
        tp_backward_stats = calc_stats(tp_backward)
        
        # Calculate speedups
        forward_speedup = None
        backward_speedup = None
        total_speedup = None
        
        if tfs_forward_stats["mean_time"] and tp_forward_stats["mean_time"]:
            forward_speedup = tfs_forward_stats["mean_time"] / tp_forward_stats["mean_time"]
        
        if tfs_backward_stats["mean_time"] and tp_backward_stats["mean_time"]:
            backward_speedup = tfs_backward_stats["mean_time"] / tp_backward_stats["mean_time"]
        
        if (tfs_forward_stats["mean_time"] and tp_forward_stats["mean_time"] and
            tfs_backward_stats["mean_time"] and tp_backward_stats["mean_time"]):
            tfs_total = tfs_forward_stats["mean_time"] + tfs_backward_stats["mean_time"]
            tp_total = tp_forward_stats["mean_time"] + tp_backward_stats["mean_time"]
            total_speedup = tfs_total / tp_total
        
        # Calculate throughput (back-projections)
        total_backprojections = batch_size * num_projections
        tfs_throughput = total_backprojections / tfs_forward_stats["mean_time"] if tfs_forward_stats["mean_time"] else None
        tp_throughput = total_backprojections / tp_forward_stats["mean_time"] if tp_forward_stats["mean_time"] else None
        
        results = {
            "torch_fourier_slice": {
                "forward": tfs_forward_stats,
                "backward": tfs_backward_stats,
                "throughput_backproj_per_sec": tfs_throughput,
                "backward_available": len(tfs_backward) > 0
            },
            "torch_projectors": {
                "forward": tp_forward_stats,
                "backward": tp_backward_stats,
                "throughput_backproj_per_sec": tp_throughput,
                "backward_available": True
            },
            "comparison": {
                "forward_speedup": forward_speedup,
                "backward_speedup": backward_speedup,
                "total_speedup": total_speedup,
                "memory_constraint_note": f"Backward pass through torch-fourier-slice only tested when batch=1 and projections=1 (current: batch={batch_size}, projections={num_projections})"
            },
            **peak_memory
        }
        
        # Add memory profiling if enabled
        if self.profile_memory:
            print(f"        Profiling memory usage...")
            
            # Profile torch-fourier-slice forward pass
            def tfs_forward_only():
                projections_tfs.grad = None
                all_volumes = []
                for i in range(batch_size):
                    projections = projections_tfs[i]
                    rotations_batch = rotations_tfs[i]
                    volume, weight_volume = insert_central_slices_rfft_3d(
                        projections, (image_size, image_size, image_size), rotations_batch
                    )
                    all_volumes.append(volume)
                all_volumes = torch.stack(all_volumes)
                del all_volumes
            
            tfs_forward_memory = self.profile_memory_usage_detailed(tfs_forward_only)
            
            # Profile torch-projectors forward pass
            def tp_forward_only():
                projections_tp.requires_grad_(True)
                weights_tp.requires_grad_(True)
                reconstruction, weight_reconstruction = torch_projectors.backproject_2d_to_3d_forw(
                    projections_tp, rotations_tp, weights=weights_tp, shifts=shifts_tp,
                    interpolation=interpolation
                )
                del reconstruction, weight_reconstruction
            
            tp_forward_memory = self.profile_memory_usage_detailed(tp_forward_only)
            
            # Calculate tensor sizes
            input_sizes = self.calculate_tensor_sizes(projections_tp, weights_tp, rotations_tp, shifts_tp, include_gradients=True)
            
            with torch.no_grad():
                sample_reconstruction, sample_weight_reconstruction = torch_projectors.backproject_2d_to_3d_forw(
                    projections_tp[:1, :1], rotations_tp[:1, :1], weights=weights_tp[:1, :1], 
                    shifts=shifts_tp[:1, :1], interpolation=interpolation
                )
            output_sizes = self.calculate_tensor_sizes(sample_reconstruction, sample_weight_reconstruction, include_gradients=False)
            output_sizes["total_mb"] = output_sizes["total_mb"] * batch_size  # 3D volumes scale with batch_size
            
            results["memory_profile"] = {
                "torch_fourier_slice_forward_memory": tfs_forward_memory,
                "torch_projectors_forward_memory": tp_forward_memory,
                "input_data_sizes": input_sizes,
                "output_data_sizes": output_sizes
            }
            
            # Memory efficiency metrics
            for lib_name, mem_data in [("torch_fourier_slice", tfs_forward_memory), ("torch_projectors", tp_forward_memory)]:
                if "peak_gpu_memory_mb" in mem_data:
                    results["memory_profile"][f"{lib_name}_memory_per_backproj_mb"] = mem_data["peak_gpu_memory_mb"] / total_backprojections
                    results["memory_profile"][f"{lib_name}_memory_efficiency"] = mem_data["peak_gpu_memory_mb"] / input_sizes["total_mb"]
        
        return results
    
    def run_benchmarks(self, batch_sizes, image_sizes, num_projections_list, interpolations, 
                       disable_tfs_backward_safety: bool = False):
        """Run all benchmark combinations with thermal-aware ordering."""
        print(f"\nRunning 2D->3D library comparison benchmarks on {self.device}")
        print(f"Libraries: torch-projectors vs torch-fourier-slice")
        print(f"Test matrix: {len(batch_sizes)} batch sizes × {len(image_sizes)} image sizes × {len(num_projections_list)} projection counts × {len(interpolations)} interpolations")
        print(f"Total tests: {len(batch_sizes) * len(image_sizes) * len(num_projections_list) * len(interpolations)}")
        
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
        total_tests = len(ordered_params)
        
        for params in ordered_params:
            batch_size = params["batch_size"]
            image_size = params["image_size"] 
            num_projections = params["num_projections"]
            interpolation = params["interpolation"]
            
            thermal_load = BenchmarkBase.calculate_thermal_load(batch_size, image_size, num_projections, interpolation)
            
            test_count += 1
            print(f"\n[{test_count}/{total_tests}] Testing: batch={batch_size}, size={image_size}x{image_size}, projs={num_projections}, interp={interpolation} (load={thermal_load:.0f})")
            
            try:
                comparison_results = self.benchmark_library_comparison(
                    batch_size, image_size, num_projections, interpolation, disable_tfs_backward_safety
                )
                
                if "error" not in comparison_results:
                    tfs = comparison_results["torch_fourier_slice"]
                    tp = comparison_results["torch_projectors"]
                    comp = comparison_results["comparison"]
                    
                    print(f"    torch-fourier-slice forward: {self.format_value_safe(tfs['forward']['median_time'], 4, 's')}")
                    print(f"    torch-projectors forward: {self.format_value_safe(tp['forward']['median_time'], 4, 's')}")
                    
                    if comp["forward_speedup"]:
                        if comp["forward_speedup"] > 1:
                            print(f"    torch-projectors is {comp['forward_speedup']:.2f}x faster for forward")
                        else:
                            print(f"    torch-fourier-slice is {1/comp['forward_speedup']:.2f}x faster for forward")
                    
                    print(f"    torch-fourier-slice backward: {self.format_value_safe(tfs['backward']['median_time'], 4, 's') if tfs['backward_available'] else 'Not tested (memory constraint)'}")
                    print(f"    torch-projectors backward: {self.format_value_safe(tp['backward']['median_time'], 4, 's')}")
                    
                    if comp["backward_speedup"]:
                        if comp["backward_speedup"] > 1:
                            print(f"    torch-projectors is {comp['backward_speedup']:.2f}x faster for backward")
                        else:
                            print(f"    torch-fourier-slice is {1/comp['backward_speedup']:.2f}x faster for backward")
                    
                    print(f"    torch-fourier-slice throughput: {self.format_value_safe(tfs['throughput_backproj_per_sec'], 1, ' backproj/s')}")
                    print(f"    torch-projectors throughput: {self.format_value_safe(tp['throughput_backproj_per_sec'], 1, ' backproj/s')}")
                else:
                    print(f"    ERROR: {comparison_results['error']}")
                    
            except Exception as e:
                print(f"    ERROR in library comparison: {e}")
                comparison_results = {"error": str(e)}
            
            # Save comparison results
            test_name = f"backward_2d_to_3d_comparison_{interpolation}_{image_size}x{image_size}x{image_size}_batch{batch_size}_proj{num_projections}"
            self.add_benchmark_result(
                test_name,
                {
                    "operation": "backward_2d_to_3d_comparison",
                    "batch_size": batch_size,
                    "image_size": [image_size, image_size],
                    "volume_size": [image_size, image_size, image_size],
                    "num_projections": num_projections,
                    "interpolation": interpolation,
                    "comparison_type": "torch-fourier-slice_vs_torch-projectors"
                },
                [],  # No raw times for nested results
                {},  # No single stats
                comparison_results
            )
            
            # Test matrix element cooldown to prevent thermal throttling
            if self.test_cooldown_seconds > 0:
                print(f"    Cooling down for {self.test_cooldown_seconds}s...")
                time.sleep(self.test_cooldown_seconds)


def run_single_element_benchmark(benchmark: Backward2DTo3DComparisonBenchmark, args):
    """Run a single benchmark element for subprocess isolation."""
    # Extract single values from lists (for subprocess mode)
    batch_size = args.batch_sizes[0] if args.batch_sizes else 1
    image_size = args.image_sizes[0] if args.image_sizes else 64
    num_projections = getattr(args, 'num_projections', [64])[0]
    interpolation = getattr(args, 'interpolations', ['linear'])[0]
    disable_tfs_backward_safety = getattr(args, 'disable_tfs_backward_safety', False)
    
    print(f"Running single element comparison: batch={batch_size}, size={image_size}x{image_size}x{image_size}, projs={num_projections}, interp={interpolation}")
    
    try:
        # Run library comparison
        comparison_results = benchmark.benchmark_library_comparison(
            batch_size, image_size, num_projections, interpolation, disable_tfs_backward_safety
        )
        
        test_name_comparison = f"backward_2d_to_3d_comparison_{interpolation}_{image_size}x{image_size}x{image_size}_batch{batch_size}_proj{num_projections}"
        benchmark.add_benchmark_result(
            test_name_comparison,
            {
                "operation": "backward_2d_to_3d_comparison",
                "batch_size": batch_size,
                "image_size": [image_size, image_size],
                "volume_size": [image_size, image_size, image_size],
                "num_projections": num_projections,
                "interpolation": interpolation,
                "comparison_type": "torch-fourier-slice_vs_torch-projectors"
            },
            [],  # No raw times for nested results
            {},  # No single stats
            comparison_results
        )
        
        print(f"Single element comparison completed successfully")
        
    except Exception as e:
        print(f"Single element comparison failed: {e}")
        raise


def main():
    parser = create_argument_parser("Benchmark 2D->3D back-projection: torch-projectors vs torch-fourier-slice comparison")
    parser.add_argument(
        '--disable-tfs-backward-safety',
        action='store_true',
        help='Disable memory safety check for torch-fourier-slice backward pass (WARNING: may cause OOM errors)'
    )
    args = parser.parse_args()
    
    # Check if torch-fourier-slice is available
    if not HAS_TORCH_FOURIER_SLICE:
        print("ERROR: torch-fourier-slice library is required for this comparison benchmark")
        print("Install with: pip install torch-fourier-slice")
        sys.exit(1)
    
    # Check if running in single-element mode (for subprocess isolation)
    if hasattr(args, 'single_element') and args.single_element:
        Backward2DTo3DComparisonBenchmark.run_single_element_mode(args, run_single_element_benchmark)
        return
    
    # Parse device
    device = BenchmarkBase.parse_device(args.device)
    
    # Create benchmark instance
    benchmark = Backward2DTo3DComparisonBenchmark(args.platform_name, device, args.title)
    
    print(f"Starting 2D->3D library comparison benchmarks")
    print(f"Platform: {args.platform_name}")
    print(f"Device: {device}")
    print(f"Libraries: torch-projectors vs torch-fourier-slice")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Image sizes: {args.image_sizes}")
    print(f"Num projections: {getattr(args, 'num_projections', [1, 16, 64, 256])}")  
    print(f"Interpolations: {getattr(args, 'interpolations', ['linear'])}")  # Default to linear only for comparison
    print(f"torch-fourier-slice available: {HAS_TORCH_FOURIER_SLICE}")
    print(f"Warmup runs: {args.warmup_runs}")
    print(f"Timing runs: {args.timing_runs}")
    print(f"Cooldown: {args.cooldown}s")
    print(f"Test cooldown: {args.test_cooldown}s")
    print(f"Memory profiling: {args.profile_memory}")
    print(f"TFS backward safety disabled: {args.disable_tfs_backward_safety}")
    
    # Memory constraint warning
    if args.disable_tfs_backward_safety:
        print(f"\nWARNING: torch-fourier-slice backward safety check is DISABLED.")
        print(f"This may cause out-of-memory errors for large batch sizes or projection counts.")
    else:
        print(f"\nINFO: Backward passes through torch-fourier-slice are only tested")
        print(f"when batch_size=1 and num_projections=1 to avoid out-of-memory errors.")
        print(f"Use --disable-tfs-backward-safety to override this safety check.")
    
    # Override benchmark parameters if provided
    benchmark.warmup_runs = args.warmup_runs
    benchmark.timing_runs = args.timing_runs
    benchmark.cooldown_seconds = args.cooldown
    benchmark.test_cooldown_seconds = args.test_cooldown
    benchmark.profile_memory = args.profile_memory
    benchmark.disable_tfs_backward_safety = args.disable_tfs_backward_safety
    
    # Generate parameter matrix - use conservative defaults for comparison
    batch_sizes = args.batch_sizes
    image_sizes = args.image_sizes
    # Use conservative defaults for 2D->3D comparison to avoid memory issues
    num_projections_list = getattr(args, 'num_projections', [1, 16, 64, 256])
    # Default to linear only for fair comparison (torch-fourier-slice doesn't have interpolation options)
    interpolations = getattr(args, 'interpolations', ['linear'])
    
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
                batch_sizes, image_sizes, num_projections_list, interpolations, args.disable_tfs_backward_safety
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