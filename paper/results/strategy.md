# Benchmarking Strategy for torch-projectors Paper

This document outlines the comprehensive benchmarking strategy for comparing `torch-projectors` with `torch-fourier-slice` across multiple platforms and configurations.

## 1. Objectives

- **Performance Comparison**: Quantify speed improvements of torch-projectors over torch-fourier-slice
- **Scalability Analysis**: Understand performance scaling with batch size and image size
- **Platform Evaluation**: Compare performance across CPU, CUDA, and MPS backends
- **Interpolation Quality vs Speed**: Analyze tradeoffs between interpolation methods
- **Memory Efficiency**: Measure memory usage patterns across different configurations

## 2. Benchmarking Framework: pyperf

### Primary Framework: pyperf
- **Advantages**: Process isolation, outlier detection, warmup handling, detailed statistics
- **Statistical rigor**: Multiple runs with proper statistical analysis
- **JSON output**: Machine-readable results for aggregation
- **Cross-platform**: Consistent methodology across different systems

### Installation Requirements
```bash
pip install pyperf psutil matplotlib seaborn pandas
```

## 3. Directory Structure

```
paper/results/
├── strategy.md                    # This document
├── benchmarks/
│   ├── benchmark_base.py          # Common utilities and base class
│   ├── forward_2d.py             # 2D forward projection benchmarks  
│   ├── backward_2d.py            # 2D backward projection benchmarks
│   ├── forward_3d.py             # 3D->2D forward benchmarks
│   ├── backward_3d.py            # 2D->3D backward benchmarks
│   ├── torch-fourier-slice.py    # torch-fourier-slice benchmarks
│   └── interpolation.py          # interpolation quality
├── data/
│   ├── forward_2d/
│   │   ├── a100-cuda.json        # Example platform results
│   │   ├── m2-mps.json
│   │   └── intel-cpu.json
│   ├── backward_2d/
│   ├── forward_3d/
│   ├── backward_3d/
│   ├── torch-fourier-slice/
│   └── interpolation/
├── plots/                        # Generated figures for paper
├── tables/                       # Generated LaTeX tables
├── aggregate_results.py          # Combine results across platforms
├── generate_plots.py            # Create publication-ready figures
└── generate_tables.py           # Create LaTeX tables
```

## 4. Benchmark Experiments

### 4.1 Core Performance Tests

#### Forward/Backward Projection Performance
- **Test Parameters**:
  - Batch sizes: [1, 8, 16]
  - Image sizes: [32, 128, 512]
  - Interpolation methods: ['linear', 'cubic']
  - Number of projections per batch: [2, 64, 256, 1024]

- **Metrics**:
  - Throughput (projections/second)
  - Peak memory usage (MB)

- **Modes to test**
  - Forward pass only with torch.no_grad()
  - Forward pass, followed by backward pass, both timed separately

### 4.3 torch-fourier-slice Comparison (`torch-fourier-slice.py`)

#### torch-projectors vs torch-fourier-slice
- torch-fourier-slice has an extremely high memory consumption in the backward pass
- Only supports batch = 1 with no additional optimizations for larger batches
- We can't test the backward pass with more than 2 poses
- For all other combinations in our matrix, we'll test forward pass only, with and without torch.no_grad()


### 4.4 Platform-Specific Tests

#### CPU Platforms
- Intel x86_64 (various generations)
- AMD x86_64
- Apple Silicon (M1/M2/M3)
- ARM64 servers

#### GPU Platforms
- NVIDIA: RTX 30/40 series, Tesla, A100, H100
- Different CUDA versions and driver combinations

#### Apple Metal (MPS)
- M1/M2/M3 variants
- Different memory configurations

## 5. Data Format Specification

### JSON Structure
```json
{
  "metadata": {
    "platform_name": "user-provided-name",
    "timestamp": "2025-01-13T10:00:00Z",
    "torch_projectors_version": "0.1.0",
    "torch_version": "2.6.0",
    "device_info": {
      "type": "cuda",
      "name": "NVIDIA A100-SXM4-80GB", 
      "memory_gb": 80,
      "compute_capability": "8.0"
    },
    "system_info": {
      "platform": "Linux-5.15.0",
      "cpu": "Intel Xeon Platinum 8358",
      "memory_gb": 512,
      "python_version": "3.11.7"
    }
  },
  "benchmarks": {
    "forward_2d_linear_128x128_batch8_proj256": {
      "parameters": {
        "operation": "forward_2d",
        "interpolation": "linear",
        "image_size": [128, 128],
        "batch_size": 8,
        "num_projections": 256,
        "warmup_runs": 5,
        "timing_runs": 20
      },
      "results": {
        "mean_time": 0.123,
        "median_time": 0.120,
        "std_dev": 0.005,
        "min_time": 0.115,
        "max_time": 0.135,
        "throughput_proj_per_sec": 16650.4,
        "peak_memory_mb": 2048.5,
        "raw_times": [0.118, 0.121, ...]
      }
    }
  }
}
```

## 6. Statistical Methodology

### Timing Protocol
1. **Warmup**: 5 runs to stabilize caches and GPU state
2. **Measurement**: 20+ timing runs for statistical significance
3. **Outlier Detection**: Remove outliers >2 standard deviations
4. **Synchronization**: Proper device synchronization (CUDA/MPS)

### Memory Measurement
- Peak GPU memory usage via `torch.cuda.max_memory_allocated()`
- System memory via `psutil.Process().memory_info()`
- Reset memory stats before each benchmark

### Error Handling
- Device availability checks
- Fallback to CPU if GPU unavailable
- Graceful handling of OOM errors
- Skip configurations that don't fit in memory

## 7. Execution Workflow

### Individual Platform Execution
```bash
# Run all benchmarks on a platform
python paper/results/benchmarks/forward_2d.py --platform-name "a100-cuda"
python paper/results/benchmarks/backward_2d.py --platform-name "a100-cuda" 
python paper/results/benchmarks/forward_3d.py --platform-name "a100-cuda"
python paper/results/benchmarks/backward_3d.py --platform-name "a100-cuda"
python paper/results/benchmarks/comparison.py --platform-name "a100-cuda"
```

### Cross-Platform Aggregation
```bash
# After collecting results from all platforms
cd paper/results
python aggregate_results.py
python generate_plots.py
python generate_tables.py
```

## 8. Output Generation

### Plots (`generate_plots.py`)
- Performance comparison bar charts
- Scaling curves (strong/weak scaling)
- Memory usage patterns
- Speedup ratios vs torch-fourier-slice
- Platform comparison matrices

### Tables (`generate_tables.py`) 
- LaTeX tables with performance numbers
- Speedup summaries
- Memory efficiency metrics
- Statistical significance tests

## 9. Quality Assurance

### Reproducibility
- Fixed random seeds for consistent test data
- Documented environment requirements
- Version pinning for critical dependencies

### Validation
- Cross-check results with existing performance tests
- Sanity checks (e.g., linear scaling expectations)
- Manual verification of key results

### Documentation
- Clear parameter documentation
- Platform-specific notes and caveats
- Troubleshooting guide for common issues

## 10. Timeline and Milestones

1. **Phase 1**: Implement base infrastructure and 2D benchmarks
2. **Phase 2**: Add 3D benchmarks and library comparison
3. **Phase 3**: Execute benchmarks across target platforms
4. **Phase 4**: Generate publication-ready results

This strategy ensures comprehensive, reproducible, and statistically sound performance evaluation of torch-projectors for the research paper.