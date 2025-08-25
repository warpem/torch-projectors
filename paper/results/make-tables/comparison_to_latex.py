#!/usr/bin/env python3
"""
Convert torch-projectors vs torch-fourier-slice comparison JSON results to LaTeX table format.

Usage:
    python comparison_to_latex.py input_folder output.tex --caption "Comparison Results"
    
Where input_folder contains multiple JSON files to be combined into a single table.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import re


def parse_comparison_benchmark_key(key):
    """Parse comparison benchmark key to extract parameters."""
    # Example: "forward_3d_to_2d_comparison_linear_32x32x32_batch1_proj8"
    # Example: "backward_2d_to_3d_comparison_linear_32x32x32_batch1_proj8"
    parts = key.split('_')
    
    # Find volume size (e.g., "32x32x32")
    volume_size = None
    for part in parts:
        if 'x' in part and len(part.split('x')) == 3:
            try:
                dims = part.split('x')
                if all(dim.isdigit() for dim in dims):
                    volume_size = int(dims[0])  # Use first dimension as box size
                    break
            except:
                continue
    
    # Find batch size
    batch_size = None
    for part in parts:
        if part.startswith('batch'):
            batch_size = int(part[5:])  # Remove 'batch' prefix
            break
    
    # Find projection count
    proj_count = None
    for part in parts:
        if part.startswith('proj'):
            proj_count = int(part[4:])  # Remove 'proj' prefix
            break
    
    return {
        'volume_size': volume_size,
        'batch_size': batch_size,
        'num_projections': proj_count
    }


def calculate_throughput_thousands(time_seconds, batch_size, num_projections):
    """Calculate throughput in thousands of projections per second."""
    if time_seconds is None or time_seconds == 0:
        return None
    total_projections = batch_size * num_projections
    throughput = total_projections / time_seconds
    return throughput / 1000.0  # Convert to thousands


def calculate_speedup(tp_time, tfs_time):
    """Calculate speedup factor (how much faster torch-projectors is)."""
    if tp_time is None or tfs_time is None or tp_time == 0:
        return None
    return tfs_time / tp_time


def format_value(value, decimal_places=1, bold=False, add_x=False):
    """Format a numeric value, using em-dash for None/null values."""
    if value is None:
        return "---"  # Em-dash
    
    # Remove decimals if value >= 10
    if value >= 10:
        formatted = f"{value:.0f}"
    else:
        formatted = f"{value:.{decimal_places}f}"
    
    # Add proper multiplication symbol prefix for speedup values
    if add_x:
        formatted = f"$\\times${formatted}"
    
    if bold:
        formatted = f"\\textbf{{{formatted}}}"
    return formatted


def organize_comparison_data(json_data_list):
    """Organize comparison benchmark data from multiple JSON files into table structure."""
    # Structure: data[volume_size][batch_size][proj_count][json_index] = {tp_forward, tp_backward, tfs_forward, tfs_backward, forward_speedup, backward_speedup}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    
    for json_index, json_info in enumerate(json_data_list):
        benchmarks = json_info['data']['benchmarks']
        
        for key, benchmark in benchmarks.items():
            # Only process comparison benchmarks (both forward and backward)
            if not (key.startswith('forward_3d_to_2d_comparison') or key.startswith('backward_2d_to_3d_comparison')):
                continue
                
            params = parse_comparison_benchmark_key(key)
            
            if not all([params['volume_size'], params['batch_size'], params['num_projections']]):
                continue
            
            volume_size = params['volume_size']
            batch_size = params['batch_size']
            proj_count = params['num_projections']
            
            results = benchmark['results']
            
            # Extract torch-projectors timing data
            tp_data = results.get('torch_projectors', {})
            tp_forward_time = tp_data.get('forward', {}).get('median_time')
            tp_backward_time = tp_data.get('backward', {}).get('median_time')
            
            # Extract torch-fourier-slice timing data
            tfs_data = results.get('torch_fourier_slice', {})
            tfs_forward_time = tfs_data.get('forward', {}).get('median_time')
            tfs_backward_time = tfs_data.get('backward', {}).get('median_time')
            
            # Calculate throughputs in thousands of projections per second
            tp_forward_throughput = calculate_throughput_thousands(tp_forward_time, batch_size, proj_count)
            tp_backward_throughput = calculate_throughput_thousands(tp_backward_time, batch_size, proj_count)
            tfs_forward_throughput = calculate_throughput_thousands(tfs_forward_time, batch_size, proj_count)
            tfs_backward_throughput = calculate_throughput_thousands(tfs_backward_time, batch_size, proj_count)
            
            # Calculate speedups
            forward_speedup = calculate_speedup(tp_forward_time, tfs_forward_time)
            backward_speedup = calculate_speedup(tp_backward_time, tfs_backward_time)
            
            data[volume_size][batch_size][proj_count][json_index] = {
                'tp_forward': tp_forward_throughput,
                'tp_backward': tp_backward_throughput,
                'tfs_forward': tfs_forward_throughput,
                'tfs_backward': tfs_backward_throughput,
                'forward_speedup': forward_speedup,
                'backward_speedup': backward_speedup
            }
    
    return data


def generate_comparison_headers(json_data_list):
    """Generate LaTeX table headers for comparison results with 3 levels."""
    # First header row: titles spanning 6 columns each (2 passes × 3 metrics)
    title_parts = []
    for json_info in json_data_list:
        title_parts.append(f"\\multicolumn{{6}}{{c}}{{{json_info['title']}}}")
    
    first_row = f"\\multirow{{3}}{{*}}[-1.2em]{{Box}} & \\multirow{{3}}{{*}}[-1.2em]{{Batch}} & \\multirow{{3}}{{*}}[-1.2em]{{Poses}} & {' & '.join(title_parts)} \\\\"
    
    # Second header row: Forward/Backward spanning 3 columns each
    pass_headers = []
    for _ in json_data_list:
        pass_headers.extend(["\\multicolumn{3}{c}{Forward}", "\\multicolumn{3}{c}{Backward}"])
    
    second_row = f"& & & {' & '.join(pass_headers)} \\\\"
    
    # Third header row: t-p, t-f-s, speedup for each pass
    metric_headers = []
    for _ in json_data_list:
        # Forward pass metrics
        metric_headers.extend(["t-f-s", "t-p", "$\\uparrow$"])
        # Backward pass metrics  
        metric_headers.extend(["t-f-s", "t-p", "$\\uparrow$"])
    
    third_row = f"& & & {' & '.join(metric_headers)} \\\\"
    
    return [first_row, second_row, third_row]


def generate_comparison_latex_table(data, json_data_list, caption=None, label=None):
    """Generate LaTeX table code for comparison results."""
    if caption is None:
        caption = "Performance Comparison: torch-projectors vs torch-fourier-slice (throughput in $10^3$ projections/second)"
    
    # Sort keys for consistent output
    volume_sizes = sorted(data.keys())
    batch_sizes = sorted(set(bs for vol_data in data.values() for bs in vol_data.keys()))
    proj_counts = sorted(set(pc for vol_data in data.values() 
                            for batch_data in vol_data.values() 
                            for pc in batch_data.keys()))
    
    # Calculate number of data columns (6 per JSON file: 2 passes × 3 metrics each)
    num_jsons = len(json_data_list)
    data_columns = num_jsons * 6
    
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    
    # Dynamic column specification: 3 fixed columns + data_columns with vertical dividers
    # Between different JSONs: full vertical divider
    # Between forward/backward within same JSON: partial vertical divider  
    json_col_spec = ""
    for i in range(num_jsons):
        if i > 0:
            json_col_spec += " | "  # Full vertical divider between different JSONs
        # Forward: t-p (0.8cm), t-f-s (0.8cm), speedup (1.3cm)
        json_col_spec += ">{\\raggedleft\\arraybackslash}p{0.8cm} >{\\raggedleft\\arraybackslash}p{0.8cm} >{\\raggedleft\\arraybackslash}p{0.8cm}"
        # Vertical divider before backward columns (within same JSON)
        json_col_spec += " | "
        # Backward: t-p (0.8cm), t-f-s (0.8cm), speedup (1.3cm)
        json_col_spec += ">{\\raggedleft\\arraybackslash}p{0.8cm} >{\\raggedleft\\arraybackslash}p{0.8cm} >{\\raggedleft\\arraybackslash}p{0.8cm}"
    
    col_spec = f"r@{{\\hspace{{3mm}}}} c@{{\\hspace{{2mm}}}} r@{{\\hspace{{3mm}}}} | {json_col_spec}"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    
    # Generate dynamic headers
    header_rows = generate_comparison_headers(json_data_list)
    lines.extend(header_rows)
    lines.append("\\midrule")
    
    for i, volume_size in enumerate(volume_sizes):
        vol_data = data[volume_size]
        # Count total rows for this volume size
        vol_total_rows = sum(len(batch_data) for batch_data in vol_data.values())
        
        first_vol_row = True
        # Get only the batch sizes that exist for this volume
        vol_batch_sizes = [bs for bs in batch_sizes if bs in vol_data]
        for j, batch_size in enumerate(vol_batch_sizes):
                
            batch_data = vol_data[batch_size]
            batch_total_rows = len(batch_data)
            
            first_batch_row = True
            proj_list = [pc for pc in proj_counts if pc in batch_data]
            for proj_idx, proj_count in enumerate(proj_list):
                proj_data = batch_data[proj_count]
                
                row_parts = []
                
                # Volume size column (multirow for entire volume)
                if first_vol_row:
                    row_parts.append(f"\\multirow{{{vol_total_rows}}}{{*}}[-1.2em]{{{volume_size}}}")
                    first_vol_row = False
                else:
                    row_parts.append("")
                
                # Batch size column (multirow for entire batch)
                if first_batch_row:
                    row_parts.append(f"\\multirow{{{batch_total_rows}}}{{*}}[-0.6em]{{{batch_size}}}")
                    first_batch_row = False
                else:
                    row_parts.append("")
                
                # Projection count column
                row_parts.append(str(proj_count))
                
                # Data columns for each JSON file (forward pass, then backward pass)
                for json_idx in range(len(json_data_list)):
                    json_data = proj_data.get(json_idx, {})
                    
                    # Forward pass metrics (swapped order: t-f-s, t-p, speedup)
                    tp_forward = json_data.get('tp_forward', None)
                    tfs_forward = json_data.get('tfs_forward', None)
                    forward_speedup = json_data.get('forward_speedup', None)
                    
                    row_parts.append(format_value(tfs_forward))
                    row_parts.append(format_value(tp_forward))
                    row_parts.append(format_value(forward_speedup, 1, bold=True, add_x=True))
                    
                    # Backward pass metrics (swapped order: t-f-s, t-p, speedup)
                    tp_backward = json_data.get('tp_backward', None)
                    tfs_backward = json_data.get('tfs_backward', None)
                    backward_speedup = json_data.get('backward_speedup', None)
                    
                    row_parts.append(format_value(tfs_backward))
                    row_parts.append(format_value(tp_backward))
                    row_parts.append(format_value(backward_speedup, 1, bold=True, add_x=True))
                
                lines.append(" & ".join(row_parts) + " \\\\")
                
                # Add horizontal separator after each row (except the last in batch)
                if proj_idx < len(proj_list) - 1:
                    # Dynamic column range: 3 (poses) to (3 + number_of_data_columns)
                    end_col = 3 + len(json_data_list) * 6
                    lines.append(f"\\cmidrule{{3-{end_col}}}")
            
            # Add cmidrule after each batch group (except the last)
            if j < len(vol_batch_sizes) - 1:
                # Dynamic column range: 2 (batch) to (3 + number_of_data_columns)
                end_col = 3 + len(json_data_list) * 6
                lines.append(f"\\cmidrule{{2-{end_col}}}")
        
        # Add midrule after each volume group (except the last)
        if i < len(volume_sizes) - 1:
            lines.append("\\midrule")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Convert comparison benchmark JSON to LaTeX table")
    parser.add_argument("input_folder", help="Input folder containing JSON files")
    parser.add_argument("output_tex", help="Output LaTeX file path")
    parser.add_argument("--caption", help="Custom table caption")
    parser.add_argument("--label", help="LaTeX table label (e.g., tab:comparison)")
    
    args = parser.parse_args()
    
    # Load all JSON files from folder
    json_files = sorted([f for f in Path(args.input_folder).glob("*.json")])
    if not json_files:
        raise ValueError(f"No JSON files found in {args.input_folder}")
    
    json_data_list = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            json_data_list.append({
                'filename': json_file.stem,
                'data': data,
                'title': data['metadata'].get('title', json_file.stem),
                'platform_name': data['metadata']['platform_name']
            })
    
    # Use first JSON's platform name (assuming they're all the same)
    platform_name = json_data_list[0]['platform_name']
    
    # Organize comparison benchmark data from all JSONs
    organized_data = organize_comparison_data(json_data_list)
    
    # Generate LaTeX table
    latex_code = generate_comparison_latex_table(organized_data, json_data_list, args.caption, args.label)
    
    # Write output
    with open(args.output_tex, 'w') as f:
        f.write(latex_code)
    
    print(f"Generated LaTeX table: {args.output_tex}")
    print(f"Combined {len(json_data_list)} JSON files:")
    for json_info in json_data_list:
        print(f"  - {json_info['filename']}: {json_info['title']}")
    print(f"Platform: {platform_name}")
    print(f"Found {len(organized_data)} volume sizes with data")


if __name__ == "__main__":
    main()