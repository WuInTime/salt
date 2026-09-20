#!/usr/bin/env python3

# MPLCONFIGDIR=/tmp/salt-matplotlib \
# python3 scripts/graph_contractions_salt_vs_cg.py --fully-db results/mlir-contractions/data-fully-associative.db --8way-db results/mlir-contractions/data-8way-associative.db --12way-db results/mlir-contractions/data-12way-associative.db --constant-dir results/mlir-contractions/work/constant --output-dir results/mlir-contractions

import argparse
import sqlite3
import json
import matplotlib

# Embed TrueType fonts in PDF output.  Matplotlib's default Type 3 glyphs are
# rejected by several camera-ready PDF validators.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTRACTION_ROOT = REPOSITORY_ROOT / 'benchmarks' / 'mlir-contractions'
DEFAULT_RESULTS_DIR = REPOSITORY_ROOT / 'results' / 'mlir-contractions'
# DEFAULT_CONSTANT_DIR = CONTRACTION_ROOT / 'constant'
DEFAULT_CONSTANT_DIR = DEFAULT_RESULTS_DIR / "work" / "constant"

def load_sqlite_data(db_path):
    """Load data from SQLite database"""
    conn = sqlite3.connect(db_path)
    # Get the first table name
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    if tables:
        table_name = tables[0][0]
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    else:
        df = pd.DataFrame()
    conn.close()
    return df

def get_json_path(program_name, base_dir='./constant/'):
    """Determine the correct JSON file path based on program name prefix"""
    # Remove ONLY the .mlir extension if present - keep constant_ prefix!
    program_name_clean = program_name.replace('.mlir', '')
    
    if program_name_clean.startswith('constant'):
        # For constant_* programs: ./constant/constant_pl-salt.json (keep constant_ prefix!)
        json_path = os.path.join(base_dir, f'{program_name_clean}-salt.json')
    elif program_name_clean.startswith('tiled'):
        # For tiled_* programs: ./constant/tiled/tiled_something-salt.json
        json_path = os.path.join(base_dir, 'tiled', f'{program_name_clean}-salt.json')
    else:
        # Default case: assume it goes in the base constant directory
        json_path = os.path.join(base_dir, f'{program_name_clean}-salt.json')
    
    return json_path

def get_display_name(program_name):
    """Get clean program name for display (remove .mlir extension, replace constant_ with orig_, remove underscores, and capitalize)"""
    clean_name = program_name.replace('.mlir', '')
    
    # Replace constant_ with orig_
    if clean_name.startswith('constant_'):
        clean_name = clean_name.replace('constant_', 'orig_')
    
    # Remove underscores and replace with spaces, then capitalize each word
    clean_name = clean_name.replace('_', ' ').title()
    
    # Handle specific case: remove redundant "Max" from "Rowwise Softmax Max" patterns
    if clean_name.endswith("Rowwise Softmax Max"):
        clean_name = clean_name.replace("Rowwise Softmax Max", "Rowwise Softmax")
    
    return clean_name

def extract_base_program_name(program_name):
    """Extract the base program name (everything after constant_ or tiled_)"""
    clean_name = program_name.replace('.mlir', '')
    if clean_name.startswith('constant_'):
        return clean_name[9:]  # Remove 'constant_'
    elif clean_name.startswith('tiled_'):
        return clean_name[6:]   # Remove 'tiled_'
    return clean_name

def organize_programs_for_layout(all_programs):
    """Organize programs into the specified 4x4 layout"""
    # Separate constant and tiled programs
    constant_programs = [p for p in all_programs if p.replace('.mlir', '').startswith('constant_')]
    tiled_programs = [p for p in all_programs if p.replace('.mlir', '').startswith('tiled_')]
    
    # Create mapping from base name to programs
    constant_map = {}
    tiled_map = {}
    
    for prog in constant_programs:
        base_name = extract_base_program_name(prog)
        constant_map[base_name] = prog
    
    for prog in tiled_programs:
        base_name = extract_base_program_name(prog)
        tiled_map[base_name] = prog
    
    # Find matching pairs (programs that have both constant and tiled versions)
    base_names = sorted(set(constant_map.keys()) & set(tiled_map.keys()))
    
    # Create the layout: 4x4 grid
    # Row 1: constant programs (positions 0-3)
    # Row 2: corresponding tiled programs (positions 4-7)
    # Row 3: more constant programs (positions 8-11)
    # Row 4: corresponding tiled programs (positions 12-15)
    
    layout = [None] * 16
    
    # Fill first 8 positions (2 rows of 4)
    for i, base_name in enumerate(base_names[:4]):
        if i < 4:
            # Row 1: constant programs
            layout[i] = constant_map[base_name]
            # Row 2: corresponding tiled programs
            layout[i + 4] = tiled_map[base_name]
    
    # Fill next 8 positions if we have more programs
    for i, base_name in enumerate(base_names[4:8]):
        if i < 4:
            # Row 3: more constant programs
            layout[i + 8] = constant_map[base_name]
            # Row 4: corresponding tiled programs
            layout[i + 12] = tiled_map[base_name]
    
    # Filter out None values and return
    return [prog for prog in layout if prog is not None]

def load_json_data(json_path):
    """Load miss ratio data from JSON file"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data['miss_ratio_curve']
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"Error loading {json_path}: {e}")
        return None

def get_total_access_for_program(program, df_8way, df_12way):
    """Get total_access for a program from 8-way or 12-way database"""
    # Try 8-way first
    if not df_8way.empty:
        program_data_8way = df_8way[df_8way['program'] == program]
        if not program_data_8way.empty:
            return program_data_8way['total_access'].iloc[0]
    
    # Try 12-way if 8-way doesn't have the program
    if not df_12way.empty:
        program_data_12way = df_12way[df_12way['program'] == program]
        if not program_data_12way.empty:
            return program_data_12way['total_access'].iloc[0]
    
    return None

def create_step_function_data(turning_points, miss_counts, extend=True):
    """Create step function data from turning points and miss counts - NO FILTERING"""
    if len(turning_points) != len(miss_counts):
        print(f"Warning: Mismatch between turning points ({len(turning_points)}) and miss counts ({len(miss_counts)})")
        return [], []
    
    if not turning_points:
        return [], []
    
    x_vals = []
    y_vals = []
    
    for i in range(len(turning_points)):
        if i == 0:
            # First point - start from the turning point
            x_vals.append(turning_points[i])
            y_vals.append(miss_counts[i])
        else:
            # Add the step: extend previous value to current turning point
            x_vals.append(turning_points[i])
            y_vals.append(miss_counts[i-1])  # Previous miss count value
            
            # Add the new value at the turning point
            x_vals.append(turning_points[i])
            y_vals.append(miss_counts[i])
    
    # Conditionally extend the last value to a reasonable end point
    if extend and len(x_vals) > 0:
        last_x = x_vals[-1]
        # Extend to 10x the last turning point or a minimum of 10000
        extend_to = max(last_x * 10, 10000)
        x_vals.append(extend_to)
        y_vals.append(y_vals[-1])
    
    return x_vals, y_vals

def clip_leading_nonpositive_data(turning_points, miss_counts):
    """Drop the initial prefix where miss counts are nonpositive."""
    first_positive_idx = next((idx for idx, count in enumerate(miss_counts) if count > 0), None)
    if first_positive_idx is None:
        return [], []
    return turning_points[first_positive_idx:], miss_counts[first_positive_idx:]

def trim_trailing_near_zero_data(turning_points, miss_ratios, zero_tol=1e-12):
    """Drop terminal near-zero SALT predictions so the final positive plateau extends flat."""
    last_positive_idx = None
    for idx, ratio in enumerate(miss_ratios):
        if ratio > zero_tol:
            last_positive_idx = idx
    if last_positive_idx is None:
        return [], []
    return turning_points[:last_positive_idx + 1], miss_ratios[:last_positive_idx + 1]

def terminal_plateau_start(turning_points, values, rel_tol=1e-9, abs_tol=1e-12):
    """Return the first turning point in the final run of equal values."""
    if not turning_points or not values:
        return None

    plateau_idx = len(values) - 1
    final_value = values[plateau_idx]
    while plateau_idx > 0 and np.isclose(
        values[plateau_idx - 1], final_value, rtol=rel_tol, atol=abs_tol
    ):
        plateau_idx -= 1
    return turning_points[plateau_idx]

def positive_values(values):
    """Return positive finite values for log-axis scaling."""
    return [value for value in values if np.isfinite(value) and value > 0]

def paired_subplot_indices(n_programs):
    """Yield constant/tiled vertical subplot pairs from the 4x4 layout."""
    for block_start in (0, 8):
        for col in range(4):
            top_idx = block_start + col
            bottom_idx = top_idx + 4
            pair = [idx for idx in (top_idx, bottom_idx) if idx < n_programs]
            if pair:
                yield pair

def plot_data(df_fully_assoc, df_8way, df_12way, constant_dir='./constant/', log_scale=True):
    """Create a single window with all programs in a 4x4 grid"""
    # Set the plotting style
    plt.style.use('ggplot')
    
    # Get all unique programs from all datasets
    programs_fully = set(df_fully_assoc['program'].unique()) if not df_fully_assoc.empty else set()
    programs_8way = set(df_8way['program'].unique()) if not df_8way.empty else set()
    programs_12way = set(df_12way['program'].unique()) if not df_12way.empty else set()
    all_programs = sorted(programs_fully.union(programs_8way).union(programs_12way))
    
    # Organize programs according to the specified layout
    organized_programs = organize_programs_for_layout(all_programs)
    
    n_programs = len(organized_programs)
    if log_scale:
        print(f"Total programs to plot: {n_programs}")
    else:
        print(f"Total programs to plot: {n_programs} - Linear scale")
    
    # Create a 4x4 grid with shorter panels to reduce empty vertical space.
    fig, axes = plt.subplots(4, 4, figsize=(24, 9.1))
    
    # Keep enough room for axis labels and the shared legend without an oversized bottom margin.
    plt.subplots_adjust(
        #     top=0.99,
        left=0,
        right=1, # Without it, Matplotlib uses the default right=0.9
        bottom=0.13, # Without it, Matplotlib uses the default bottom=0.1
        wspace=0.2, hspace=1.0)
    
    # Flatten axes array for easier indexing
    axes_flat = axes.flatten()
    
    # Keep track of legend handles and labels (we'll create the legend once)
    legend_handles = []
    legend_labels = []
    legend_created = False
    x_limit_values_by_idx = {}
    measured_x_values_by_idx = {}
    salt_trim_plateau_start_by_idx = {}
    
    for idx, program in enumerate(organized_programs):
        if idx >= 16:  # Safety check - only plot first 16 programs
            print(f"Warning: More than 16 programs found. Only plotting first 16.")
            break
            
        ax = axes_flat[idx]
        display_name = get_display_name(program)
        positive_y_values = []
        x_limit_values = []
        measured_x_values = []
        salt_trim_plateau_start = None
        
        # Plot fully associative data with diamond markers (using miss counts) - NO FILTER, NO EXTEND
        program_data_fully = df_fully_assoc[df_fully_assoc['program'] == program]
        if not program_data_fully.empty:
            program_data_fully = program_data_fully.copy()
            program_data_fully = program_data_fully.sort_values('d1_associativity')
            
            # Create step function data for fully associative after clipping the leading nonpositive prefix.
            turning_points = program_data_fully['d1_associativity'].tolist()
            miss_counts = program_data_fully['d1_miss_count'].tolist()
            turning_points, miss_counts = clip_leading_nonpositive_data(turning_points, miss_counts)
            step_x, step_y = create_step_function_data(turning_points, miss_counts, extend=False)
            fully_x_values = positive_values(step_x)
            x_limit_values.extend(fully_x_values)
            measured_x_values.extend(fully_x_values)
            positive_y_values.extend(positive_values(step_y))
            
            if step_x and step_y:
                line1, = ax.plot(step_x, step_y, 
                'D', label='Fully Assoc',
                linewidth=1, markersize=2.5, color='gray')
                if not legend_created:
                    legend_handles.append(line1)
                    legend_labels.append('Fully Assoc')
        
        # Plot 8-way associative data with X marks (using miss counts) - NO FILTER
        program_data_8way = df_8way[df_8way['program'] == program]
        if not program_data_8way.empty:
            program_data_8way = program_data_8way.copy()
            program_data_8way['cache_size'] = program_data_8way['d1_cache_size'] / program_data_8way['d1_block_size']
            program_data_8way = program_data_8way.sort_values('cache_size')
            
            line2, = ax.plot(program_data_8way['cache_size'], program_data_8way['d1_miss_count'], 
                            'x', label='8-way', 
                            markersize=12, markeredgewidth=2)  # Removed color='red'
            cache_x_values = positive_values(program_data_8way['cache_size'])
            x_limit_values.extend(cache_x_values)
            measured_x_values.extend(cache_x_values)
            positive_y_values.extend(positive_values(program_data_8way['d1_miss_count']))
            if not legend_created:
                legend_handles.append(line2)
                legend_labels.append('8-way')
        
        # Plot 12-way associative data with + marks (using miss counts) - NO FILTER
        program_data_12way = df_12way[df_12way['program'] == program]
        if not program_data_12way.empty:
            program_data_12way = program_data_12way.copy()
            program_data_12way['cache_size'] = program_data_12way['d1_cache_size'] / program_data_12way['d1_block_size']
            program_data_12way = program_data_12way.sort_values('cache_size')
            
            line3, = ax.plot(program_data_12way['cache_size'], program_data_12way['d1_miss_count'], 
                            '+', label='12-way', 
                            markersize=12, markeredgewidth=2)  # Removed color='green'
            cache_x_values = positive_values(program_data_12way['cache_size'])
            x_limit_values.extend(cache_x_values)
            measured_x_values.extend(cache_x_values)
            positive_y_values.extend(positive_values(program_data_12way['d1_miss_count']))
            if not legend_created:
                legend_handles.append(line3)
                legend_labels.append('12-way')
        
        # Get the appropriate JSON path and plot SALT data (convert to miss counts) - NO FILTER, WITH EXTEND
        json_path = get_json_path(program, constant_dir)
        json_data = load_json_data(json_path)
        if json_data:
            miss_ratios = json_data['miss_ratio']
            turning_points = json_data['turning_points']
            
            # Get total_access for this program to convert ratios to counts
            total_access = get_total_access_for_program(program, df_8way, df_12way)
            
            if total_access is not None:
                turning_points, miss_ratios = trim_trailing_near_zero_data(turning_points, miss_ratios)
                salt_turning_points = positive_values(turning_points)
                x_limit_values.extend(salt_turning_points)
                if salt_turning_points:
                    salt_trim_plateau_start = terminal_plateau_start(turning_points, miss_ratios)

                # Convert miss ratios to miss counts
                miss_counts = [ratio * total_access for ratio in miss_ratios]
                
                # Create step function data - NO FILTER, WITH EXTEND (default)
                step_x, step_y = create_step_function_data(turning_points, miss_counts, extend=True)
                positive_y_values.extend(positive_values(step_y))
                
                if step_x and step_y:
                    # Plot JSON data as step function
                    line4, = ax.plot(step_x, step_y, 
                        '--', label='SALT', 
                        linewidth=2, alpha=0.8, color='blue')  # Made SALT blue
                    if not legend_created:
                        legend_handles.append(line4)
                        legend_labels.append('SALT')
            else:
                print(f"Warning: Could not find total_access for program {program}, skipping SALT data")
        
        # Mark that we've collected legend info from the first subplot
        if not legend_created and legend_handles:
            legend_created = True

        x_limit_values_by_idx[idx] = x_limit_values
        measured_x_values_by_idx[idx] = measured_x_values
        salt_trim_plateau_start_by_idx[idx] = salt_trim_plateau_start
        
        # Set x scale (always log) and y scale based on parameter
        ax.set_xscale('log')
        if log_scale:
            ax.set_yscale('log')
        else:
            ax.set_yscale('linear')
        
        # Customize the subplot with appropriate font sizes (doubled)
        ax.set_xlabel('Cache Size', fontsize=18, loc='right')
        y_label = 'Miss Count'
        ax.set_ylabel(y_label, fontsize=18)
        ax.set_title(display_name, fontsize=20, pad=8, fontweight='bold')
        
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # NO x-axis limit filtering - let matplotlib auto-scale based on data
        
        # Set y-axis limits based on scale type
        if log_scale:
            if positive_y_values:
                min_y = min(positive_y_values)
                max_y = max(positive_y_values)
                ax.set_ylim(bottom=min_y / 1.5, top=max_y * 1.5)
        else:
            ax.set_ylim(bottom=0)  # Start from 0 for linear scale
        
        # Larger tick marks and doubled tick label font sizes
        ax.tick_params(axis='both', which='major', labelsize=16, length=8, width=2)
        ax.tick_params(axis='both', which='minor', labelsize=12, length=4, width=1)

    # Share x-ranges within each constant/tiled pair while excluding artificial SALT extensions.
    for pair in paired_subplot_indices(n_programs):
        pair_x_values = []
        log_right_candidates = []
        for idx in pair:
            pair_x_values.extend(x_limit_values_by_idx.get(idx, []))
            measured_x_values = measured_x_values_by_idx.get(idx, [])
            salt_trim_plateau_start = salt_trim_plateau_start_by_idx.get(idx)
            if salt_trim_plateau_start is not None:
                last_measured_x = max(measured_x_values) if measured_x_values else 0
                if last_measured_x >= salt_trim_plateau_start:
                    log_right_candidates.append(last_measured_x * 2.0)
                else:
                    log_right_candidates.append(salt_trim_plateau_start * 1.25)
            elif measured_x_values:
                log_right_candidates.append(max(measured_x_values) * 2.0)
        if pair_x_values:
            min_x = min(pair_x_values)
            max_x = max(pair_x_values)
            if min_x == max_x:
                left, right = min_x / 1.5, max_x * 1.5
            else:
                left, right = min_x / 1.25, max_x * 1.25
            if log_scale and log_right_candidates:
                right = max(log_right_candidates)
            for idx in pair:
                axes_flat[idx].set_xlim(left=left, right=right)
    
    # Hide any unused subplots
    for idx in range(n_programs, 16):
        axes_flat[idx].set_visible(False)
    
    # Create a single legend positioned at the bottom
    if legend_handles:
        fig.legend(legend_handles, legend_labels, 
                  loc='lower right',
                #   bbox_to_anchor=(0.96, -0.12),
                  borderaxespad=0.1,
                  ncol=len(legend_labels), fontsize=18,
                  frameon=True, fancybox=True, shadow=True)
    
    return fig

def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot Cachegrind miss counts and SALT predictions for MLIR contractions.'
    )
    parser.add_argument(
        '--fully-db', type=Path,
        default=DEFAULT_RESULTS_DIR / 'data-fully-associative.db',
        help='fully associative Cachegrind database',
    )
    parser.add_argument(
        '--8way-db', dest='db_8way', type=Path,
        default=DEFAULT_RESULTS_DIR / 'data-8way-associative.db',
        help='8-way Cachegrind database',
    )
    parser.add_argument(
        '--12way-db', dest='db_12way', type=Path,
        default=DEFAULT_RESULTS_DIR / 'data-12way-associative.db',
        help='12-way Cachegrind database',
    )
    parser.add_argument(
        '--constant-dir', type=Path, default=DEFAULT_CONSTANT_DIR,
        help='directory containing constant and tiled SALT JSON files',
    )
    parser.add_argument(
        '--output-dir', type=Path, default=DEFAULT_RESULTS_DIR,
        help='directory for generated SVG files',
    )
    parser.add_argument(
        '--show', action='store_true',
        help='open interactive plot windows after saving the SVG files',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    db_path_fully = args.fully_db
    db_path_8way = args.db_8way
    db_path_12way = args.db_12way
    constant_dir = args.constant_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load fully associative data
    print("Loading fully associative data...")
    if os.path.exists(db_path_fully):
        df_fully_assoc = load_sqlite_data(db_path_fully)
        print(f"Loaded {len(df_fully_assoc)} rows from fully associative DB")
        if not df_fully_assoc.empty:
            print("Fully associative columns:", df_fully_assoc.columns.tolist())
    else:
        print(f"Database {db_path_fully} not found!")
        df_fully_assoc = pd.DataFrame()
    
    # Load 8-way associative data
    print("Loading 8-way associative data...")
    if os.path.exists(db_path_8way):
        df_8way = load_sqlite_data(db_path_8way)
        print(f"Loaded {len(df_8way)} rows from 8-way associative DB")
        if not df_8way.empty:
            print("8-way associative columns:", df_8way.columns.tolist())
    else:
        print(f"Database {db_path_8way} not found!")
        df_8way = pd.DataFrame()
    
    # Load 12-way associative data
    print("Loading 12-way associative data...")
    if os.path.exists(db_path_12way):
        df_12way = load_sqlite_data(db_path_12way)
        print(f"Loaded {len(df_12way)} rows from 12-way associative DB")
        if not df_12way.empty:
            print("12-way associative columns:", df_12way.columns.tolist())
    else:
        print(f"Database {db_path_12way} not found!")
        df_12way = pd.DataFrame()
    
    if df_fully_assoc.empty and df_8way.empty and df_12way.empty:
        print("No data found in any database!")
        return
    
    # Get all unique programs and show the layout
    programs_fully = set(df_fully_assoc['program'].unique()) if not df_fully_assoc.empty else set()
    programs_8way = set(df_8way['program'].unique()) if not df_8way.empty else set()
    programs_12way = set(df_12way['program'].unique()) if not df_12way.empty else set()
    all_programs = sorted(programs_fully.union(programs_8way).union(programs_12way))
    
    organized_programs = organize_programs_for_layout(all_programs)
    
    print(f"Programs found: {[get_display_name(p) for p in all_programs]}")
    print(f"Layout: {[get_display_name(p) for p in organized_programs]}")
    
    # Check required columns for each dataset
    if not df_fully_assoc.empty:
        required_columns_fully = ['program', 'd1_associativity', 'd1_miss_count', 'total_access']
        missing_columns_fully = [col for col in required_columns_fully if col not in df_fully_assoc.columns]
        if missing_columns_fully:
            print(f"Missing required columns in fully associative DB: {missing_columns_fully}")
    
    if not df_8way.empty:
        required_columns_8way = ['program', 'd1_cache_size', 'd1_block_size', 'd1_miss_count', 'total_access']
        missing_columns_8way = [col for col in required_columns_8way if col not in df_8way.columns]
        if missing_columns_8way:
            print(f"Missing required columns in 8-way associative DB: {missing_columns_8way}")
    
    if not df_12way.empty:
        required_columns_12way = ['program', 'd1_cache_size', 'd1_block_size', 'd1_miss_count', 'total_access']
        missing_columns_12way = [col for col in required_columns_12way if col not in df_12way.columns]
        if missing_columns_12way:
            print(f"Missing required columns in 12-way associative DB: {missing_columns_12way}")
    
    # Show which JSON files we'll be looking for (with examples)
    print("\nJSON files to look for:")
    for program in organized_programs:
        json_path = get_json_path(program, constant_dir)
        exists = "✓" if os.path.exists(json_path) else "✗"
        in_fully = "✓" if program in programs_fully else "✗"
        in_8way = "✓" if program in programs_8way else "✗"
        in_12way = "✓" if program in programs_12way else "✗"
        total_access = get_total_access_for_program(program, df_8way, df_12way)
        total_access_str = f"total_access: {total_access}" if total_access else "no total_access found"
        print(f"  {exists} {json_path} (for {get_display_name(program)}) [Fully: {in_fully}, 8-way: {in_8way}, 12-way: {in_12way}] ({total_access_str})")
    
    # Create plots with both log and linear scales
    print(f"\nCreating plots with all {len(organized_programs)} programs (NO FILTERING)...")
    
    # LOG SCALE PLOT (implicit - no mention of log scaling)
    print("Creating plot...")
    fig_log = plot_data(df_fully_assoc, df_8way, df_12way, constant_dir, log_scale=True)
    
    # Save log scale plot as SVG and PDF
    filename_log = args.output_dir / 'miss_count_comparison_all_programs_log.svg'
    fig_log.savefig(filename_log, format='svg', bbox_inches='tight', facecolor='white', edgecolor='none')
    filename_log_pdf = filename_log.with_suffix('.pdf')
    fig_log.savefig(filename_log_pdf, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Plots saved as '{filename_log}' and '{filename_log_pdf}' (vector formats)")
    
    # LINEAR SCALE PLOT
    print("Creating linear scale plot...")
    fig_linear = plot_data(df_fully_assoc, df_8way, df_12way, constant_dir, log_scale=False)
    
    # Save linear scale plot as SVG and PDF
    filename_linear = args.output_dir / 'miss_count_comparison_all_programs_linear.svg'
    fig_linear.savefig(filename_linear, format='svg', bbox_inches='tight', facecolor='white', edgecolor='none')
    filename_linear_pdf = filename_linear.with_suffix('.pdf')
    fig_linear.savefig(filename_linear_pdf, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Linear scale plots saved as '{filename_linear}' and '{filename_linear_pdf}' (vector formats)")
    
    if args.show:
        plt.show()
    else:
        plt.close(fig_log)
        plt.close(fig_linear)

if __name__ == "__main__":
    main()
