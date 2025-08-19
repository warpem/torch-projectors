#!/usr/bin/env python3
"""
Convert benchmark JSON results to LaTeX table format.

Usage:
    python json_to_latex.py input_folder output.tex --caption "Performance Results"
    
Where input_folder contains multiple JSON files to be combined into a single table.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def parse_benchmark_key(key):
    """Parse benchmark key to extract parameters."""
    # Example: "forward_2d_linear_128x128_batch1_proj64_grad"
    parts = key.split('_')
    
    # Find interpolation method (linear/cubic)
    interp = None
    for part in parts:
        if part in ['linear', 'cubic']:
            interp = part
            break
    
    # Find image size (e.g., "128x128")
    image_size = None
    for part in parts:
        if 'x' in part and part.replace('x', '').replace('128', '').replace('256', '').replace('512', '').replace('32', '') == '':
            image_size = int(part.split('x')[0])
            break
    
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
    
    # Check if gradients are enabled
    has_gradients = 'grad' in parts
    
    return {
        'interpolation': interp,
        'image_size': image_size,
        'batch_size': batch_size,
        'num_projections': proj_count,
        'with_gradients': has_gradients
    }


def calculate_throughput(time_seconds, batch_size, num_projections):
    """Calculate throughput in thousands of projections per second."""
    if time_seconds is None or time_seconds == 0:
        return None
    total_projections = batch_size * num_projections
    throughput = total_projections / time_seconds
    return throughput / 1000.0  # Convert to thousands


def format_value(value, decimal_places=1):
    """Format a numeric value, using em-dash for None/null values."""
    if value is None:
        return "---"  # Em-dash
    return f"{value:.{decimal_places}f}"


def organize_data(benchmarks):
    """Organize benchmark data into the table structure."""
    # Structure: data[box_size][batch_size][proj_count][interpolation] = {forward, backward}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    
    for key, benchmark in benchmarks.items():
        params = parse_benchmark_key(key)
        
        if not all([params['interpolation'], params['image_size'], 
                   params['batch_size'], params['num_projections']]):
            continue
            
        if not params['with_gradients']:
            continue  # We only want gradient-enabled benchmarks for backward pass
        
        box_size = params['image_size']
        batch_size = params['batch_size']
        proj_count = params['num_projections']
        interp = params['interpolation']
        
        results = benchmark['results']
        
        # Calculate throughputs
        forward_time = results['forward']['median_time']
        backward_time = results['backward']['median_time']
        
        forward_throughput = calculate_throughput(forward_time, batch_size, proj_count)
        backward_throughput = calculate_throughput(backward_time, batch_size, proj_count)
        
        data[box_size][batch_size][proj_count][interp] = {
            'forward': forward_throughput,
            'backward': backward_throughput
        }
    
    return data


def organize_multi_json_data(json_data_list):
    """Organize benchmark data from multiple JSON files into the table structure."""
    # Structure: data[box_size][batch_size][proj_count][interpolation][json_index] = {forward, backward}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))
    
    for json_index, json_info in enumerate(json_data_list):
        benchmarks = json_info['data']['benchmarks']
        
        for key, benchmark in benchmarks.items():
            params = parse_benchmark_key(key)
            
            if not all([params['interpolation'], params['image_size'], 
                       params['batch_size'], params['num_projections']]):
                continue
                
            if not params['with_gradients']:
                continue  # We only want gradient-enabled benchmarks for backward pass
            
            box_size = params['image_size']
            batch_size = params['batch_size']
            proj_count = params['num_projections']
            interp = params['interpolation']
            
            results = benchmark['results']
            
            # Calculate throughputs
            forward_time = results['forward']['median_time']
            backward_time = results['backward']['median_time']
            
            forward_throughput = calculate_throughput(forward_time, batch_size, proj_count)
            backward_throughput = calculate_throughput(backward_time, batch_size, proj_count)
            
            data[box_size][batch_size][proj_count][interp][json_index] = {
                'forward': forward_throughput,
                'backward': backward_throughput
            }
    
    return data


def generate_multi_json_headers(json_data_list):
    """Generate LaTeX table headers for multiple JSON files."""
    # First header row: titles spanning 2 columns each
    title_parts = []
    for json_info in json_data_list:
        title_parts.append(f"\\multicolumn{{2}}{{c}}{{{json_info['title']}}}")
    
    first_row = f"\\multirow{{2}}{{*}}[-0.6em]{{Box}} & \\multirow{{2}}{{*}}[-0.6em]{{Batch}} & \\multirow{{2}}{{*}}[-0.6em]{{Poses}} & \\multirow{{2}}{{*}}[-0.6em]{{Interp.}} & {' & '.join(title_parts)} \\\\"
    
    # Second header row: Forward and Forward+Backward columns
    column_headers = []
    for _ in json_data_list:
        column_headers.extend(["Fwd", "Fwd+Bwd"])
    
    second_row = f"& & & & {' & '.join(column_headers)} \\\\"
    
    return [first_row, second_row]


def generate_latex_table(data, json_data_list, caption=None, label=None):
    """Generate LaTeX table code."""
    if caption is None:
        caption = "Performance Results"
    
    # Sort keys for consistent output
    box_sizes = sorted(data.keys())
    batch_sizes = sorted(set(bs for box_data in data.values() for bs in box_data.keys()))
    proj_counts = sorted(set(pc for box_data in data.values() 
                            for batch_data in box_data.values() 
                            for pc in batch_data.keys()))
    
    # Calculate number of data columns (2 per JSON file: Forward, Forward+Backward)
    num_jsons = len(json_data_list)
    data_columns = num_jsons * 2
    
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    
    # Dynamic column specification: 4 fixed columns + data_columns
    col_spec = f"r@{{\\hspace{{3mm}}}} c@{{\\hspace{{2mm}}}} r@{{\\hspace{{3mm}}}} c@{{\\hspace{{2mm}}}} | *{{{data_columns}}}{{>{{\\raggedleft\\arraybackslash}}p{{1.5cm}}}}"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    
    # Generate dynamic headers
    header_rows = generate_multi_json_headers(json_data_list)
    lines.extend(header_rows)
    lines.append("\\midrule")
    
    for i, box_size in enumerate(box_sizes):
        box_data = data[box_size]
        # Count total rows: each proj_count gets 2 rows (linear + cubic)
        box_total_rows = sum(len(batch_data) * 2 for batch_data in box_data.values())
        
        first_box_row = True
        for j, batch_size in enumerate(batch_sizes):
            if batch_size not in box_data:
                continue
                
            batch_data = box_data[batch_size]
            batch_total_rows = len(batch_data) * 2  # 2 rows per proj_count
            
            first_batch_row = True
            proj_list = [pc for pc in proj_counts if pc in batch_data]
            for proj_idx, proj_count in enumerate(proj_list):
                proj_data = batch_data[proj_count]
                
                # Create two rows: one for Linear, one for Cubic
                for interp_idx, interp_name in enumerate(['Linear', 'Cubic']):
                    row_parts = []
                    
                    # Box size column (multirow for entire box)
                    if first_box_row:
                        row_parts.append(f"\\multirow{{{box_total_rows}}}{{*}}[-1.2em]{{{box_size}}}")
                        first_box_row = False
                    else:
                        row_parts.append("")
                    
                    # Batch size column (multirow for entire batch)
                    if first_batch_row:
                        row_parts.append(f"\\multirow{{{batch_total_rows}}}{{*}}[-0.6em]{{{batch_size}}}")
                        first_batch_row = False
                    else:
                        row_parts.append("")
                    
                    # Poses column (multirow for both Linear and Cubic rows)
                    if interp_idx == 0:  # First row (Linear)
                        row_parts.append(f"\\multirow{{2}}{{*}}[-0.1em]{{{proj_count}}}")
                    else:  # Second row (Cubic)
                        row_parts.append("")
                    
                    # Interpolation column
                    row_parts.append(interp_name)
                    
                    # Data columns for each JSON file
                    for json_idx in range(len(json_data_list)):
                        interp_lower = interp_name.lower()
                        json_data = proj_data.get(interp_lower, {}).get(json_idx, {})
                        forward_val = json_data.get('forward', None)
                        backward_val = json_data.get('backward', None)
                        
                        row_parts.append(format_value(forward_val))
                        row_parts.append(format_value(backward_val))
                    
                    lines.append(" & ".join(row_parts) + " \\\\")
                    
                    # Add horizontal separator after each cubic row (end of each pair)
                    # But only if it's not the last proj in the last batch of a box
                    if interp_idx == 1:  # After Cubic row
                        is_last_box = i == len(box_sizes) - 1
                        is_last_batch_in_box = j == len([bs for bs in batch_sizes if bs in box_data]) - 1
                        is_last_proj_in_batch = proj_idx == len(proj_list) - 1
                        is_last_row_in_table = is_last_box and is_last_batch_in_box and is_last_proj_in_batch
                        will_have_box_separator = is_last_batch_in_box and i < len(box_sizes) - 1
                        
                        # Don't add cmidrule if we're at the very last row of the table OR
                        # if we're at the last proj in last batch and there will be a box separator  
                        if not (is_last_row_in_table or (is_last_proj_in_batch and will_have_box_separator)):
                            # Dynamic column range: 3 (poses) to (4 + number_of_data_columns)
                            end_col = 4 + len(json_data_list) * 2
                            lines.append(f"\\cmidrule{{3-{end_col}}}")
            
            # Add cmidrule after each batch group (except the last)
            if j < len([bs for bs in batch_sizes if bs in box_data]) - 1:
                # Dynamic column range: 2 (batch) to (4 + number_of_data_columns)
                end_col = 4 + len(json_data_list) * 2
                lines.append(f"\\cmidrule{{2-{end_col}}}")
        
        # Add midrule after each box group (except the last)
        if i < len(box_sizes) - 1:
            lines.append("\\midrule")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Convert benchmark JSON to LaTeX table")
    parser.add_argument("input_folder", help="Input folder containing JSON files")
    parser.add_argument("output_tex", help="Output LaTeX file path")
    parser.add_argument("--caption", help="Custom table caption")
    parser.add_argument("--label", help="LaTeX table label (e.g., tab:performance)")
    
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
    
    # Organize benchmark data from all JSONs
    organized_data = organize_multi_json_data(json_data_list)
    
    # Generate LaTeX table
    latex_code = generate_latex_table(organized_data, json_data_list, args.caption, args.label)
    
    # Write output
    with open(args.output_tex, 'w') as f:
        f.write(latex_code)
    
    print(f"Generated LaTeX table: {args.output_tex}")
    print(f"Combined {len(json_data_list)} JSON files:")
    for json_info in json_data_list:
        print(f"  - {json_info['filename']}: {json_info['title']}")
    print(f"Platform: {platform_name}")
    print(f"Found {len(organized_data)} box sizes with data")


if __name__ == "__main__":
    main()