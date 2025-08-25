#!/usr/bin/env python3
"""
Generate LaTeX table from oversampling benchmark JSON files.

Usage:
    python oversampling_table_generator.py input_folder output.tex --caption "Performance Results"
"""

import json
import argparse
from pathlib import Path


def load_json_files(folder_path):
    """Load all JSON files from the specified folder."""
    folder = Path(folder_path)
    json_files = []
    
    for json_file in folder.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                json_files.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return json_files


def extract_oversampling_data(json_data):
    """Extract oversampling benchmark data from a single JSON file."""
    platform = json_data['metadata']['title']
    benchmarks = json_data['benchmarks']
    
    # Map scenario descriptions to our table columns
    scenario_mapping = {
        '128x128 basic': 'oversampling_1',
        '256x256 oversampling=2': 'oversampling_2', 
        'real-space padding + oversampling=2': 'pad_2x_oversampling_2'
    }
    
    result = {
        'platform': platform,
        'oversampling_1': {'linear': None, 'cubic': None},
        'oversampling_2': {'linear': None, 'cubic': None},
        'pad_2x_oversampling_2': {'linear': None, 'cubic': None}
    }
    
    for benchmark_key, benchmark_data in benchmarks.items():
        scenario_desc = benchmark_data['parameters']['scenario_description']
        interpolation = benchmark_data['parameters']['interpolation']
        
        if scenario_desc in scenario_mapping:
            scenario_key = scenario_mapping[scenario_desc]
            # Use the pre-calculated throughput_forward_proj_per_sec value
            throughput = benchmark_data['results']['throughput_forward_proj_per_sec']
            result[scenario_key][interpolation] = throughput
    
    return result


def format_value(value, decimal_places=1):
    """Format a numeric value, using em-dash for None/null values."""
    if value is None:
        return "---"
    # Convert to thousands and format with decimal places
    return f"{value / 1000:.{decimal_places}f}"


def generate_latex_table(platform_data, caption=None):
    """Generate LaTeX table code for oversampling comparison."""
    if caption is None:
        caption = "Projecting 4096 128x128px references, throughput in $10^3$ projections/second"
    
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\label{tab:oversampling}")
    lines.append("\\begin{tabular}{ll|rrr}")
    lines.append("\\toprule")
    lines.append("Platform & Interpolation & Oversampling = 1.0 & Oversampling = 2.0 & Pad 2x, Oversampling = 2.0 \\\\")
    lines.append("\\midrule")
    
    for i, data in enumerate(platform_data):
        platform = data['platform']
        
        # First row: bi-linear
        linear_vals = [
            format_value(data['oversampling_1']['linear']),
            format_value(data['oversampling_2']['linear']),
            format_value(data['pad_2x_oversampling_2']['linear'])
        ]
        
        # Use multirow for platform name spanning 2 rows
        lines.append(f"\\multirow{{2}}{{*}}{{{platform}}} & linear & {' & '.join(linear_vals)} \\\\")
        
        # Second row: bi-cubic
        cubic_vals = [
            format_value(data['oversampling_1']['cubic']),
            format_value(data['oversampling_2']['cubic']),
            format_value(data['pad_2x_oversampling_2']['cubic'])
        ]
        lines.append(f"& cubic & {' & '.join(cubic_vals)} \\\\")
        
        # Add midrule between platforms (except after the last one)
        if i < len(platform_data) - 1:
            lines.append("\\midrule")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX table from oversampling benchmark JSONs")
    parser.add_argument("input_folder", help="Input folder containing JSON files")
    parser.add_argument("output_tex", help="Output LaTeX file path")
    parser.add_argument("--caption", help="Custom table caption")
    
    args = parser.parse_args()
    
    # Load all JSON files from the folder
    json_files = load_json_files(args.input_folder)
    
    if not json_files:
        print("No valid JSON files found in the specified folder.")
        return
    
    # Extract data from each JSON file
    platform_data = []
    for json_data in json_files:
        data = extract_oversampling_data(json_data)
        platform_data.append(data)
    
    # Sort by platform name for consistent output
    platform_data.sort(key=lambda x: x['platform'])
    
    # Generate LaTeX table
    latex_code = generate_latex_table(platform_data, args.caption)
    
    # Write output
    with open(args.output_tex, 'w') as f:
        f.write(latex_code)
    
    print(f"Generated LaTeX table: {args.output_tex}")
    print(f"Found {len(platform_data)} platforms")
    for data in platform_data:
        print(f"  - {data['platform']}")


if __name__ == "__main__":
    main()