#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPOSITORY_ROOT / 'analyzer' / 'misc' / 'benchmark' / 'results' / 'matmul-t0-t2'

# Set the plot style to ggplot
plt.style.use('ggplot')

def binary_search_interp_vectorized(matmul_blocks, salt_turning_points, salt_miss_ratio):
    # Convert inputs to numpy arrays if they aren't already
    matmul_blocks = np.asarray(matmul_blocks)
    salt_turning_points = np.asarray(salt_turning_points)
    salt_miss_ratio = np.asarray(salt_miss_ratio)
    
    # Initialize result array
    result = np.zeros_like(matmul_blocks)
    
    # Handle empty turning points case
    if len(salt_turning_points) == 0:
        return result
    
    # Get the indices for each block
    # Using searchsorted which is numpy's equivalent of bisect
    indices = np.searchsorted(salt_turning_points, matmul_blocks, side='right') - 1
    
    # Handle values before first turning point
    mask_before = matmul_blocks < salt_turning_points[0]
    result[mask_before] = 1.0
    
    # Handle values after last turning point
    mask_after = matmul_blocks >= salt_turning_points[-1]
    result[mask_after] = 0.0
    
    # Handle values in between
    mask_middle = ~mask_before & ~mask_after
    valid_indices = indices[mask_middle]
    # Ensure indices are within bounds (should be already due to masks)
    valid_indices = np.clip(valid_indices, 0, len(salt_miss_ratio) - 1)
    result[mask_middle] = salt_miss_ratio[valid_indices]
    
    return result


def merge_additional_runs(matmul_blocks, matmul_miss_ratio, tmp_file):
    """Merge additional runs from `tmp_file` into the base arrays.
    If `tmp_file` doesn't exist or has no data, return originals.
    For duplicate block sizes, the miss ratios are averaged.
    Returns (blocks_sorted_unique, miss_ratio_for_each_block).
    """
    if not os.path.exists(tmp_file):
        return matmul_blocks, matmul_miss_ratio

    try:
        with open(tmp_file) as f:
            extra = json.load(f)
    except Exception:
        return matmul_blocks, matmul_miss_ratio

    extra_blocks = np.array(extra.get('blocks', []))
    extra_miss = np.array(extra.get('miss_ratio', []))

    if extra_blocks.size == 0:
        return matmul_blocks, matmul_miss_ratio

    all_blocks = np.concatenate([np.asarray(matmul_blocks), extra_blocks])
    all_miss = np.concatenate([np.asarray(matmul_miss_ratio), extra_miss])

    # Group by block value and average miss ratios for duplicates. np.unique sorts by value.
    uniq_blocks, inv = np.unique(all_blocks, return_inverse=True)
    mean_miss = np.zeros_like(uniq_blocks, dtype=float)
    for i in range(len(uniq_blocks)):
        mean_miss[i] = all_miss[inv == i].mean()

    return uniq_blocks, mean_miss


def autoscale_visible_log_y(
    ax, x_min, x_max, padding_decades=0.12, numerical_zero=1e-12
):
    """Set log-y limits from nonzero values inside the visible x range."""
    visible_y = []

    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        mask = (x >= x_min) & (x <= x_max) & np.isfinite(y) & (y > numerical_zero)
        visible_y.extend(y[mask])

    for collection in ax.collections:
        offsets = np.asarray(collection.get_offsets(), dtype=float)
        if offsets.ndim != 2 or offsets.shape[1] < 2:
            continue
        x = offsets[:, 0]
        y = offsets[:, 1]
        mask = (x >= x_min) & (x <= x_max) & np.isfinite(y) & (y > numerical_zero)
        visible_y.extend(y[mask])

    if visible_y:
        log_y = np.log10(np.asarray(visible_y))
        lower = log_y.min()
        upper = log_y.max()
        padding = max((upper - lower) * padding_decades, 0.08)
        ax.set_ylim(10 ** (lower - padding), 10 ** (upper + padding))

def run(svg_output, data_dir):
    # Define the file groups with ggplot-friendly colors
    groups = [
        {
            'matmul_file': data_dir / 'matmul.json',
            'salt_file': data_dir / 'matmul-salt.json',
            'label': 'No Tiling',
            'color': '#1f77b4',  # ggplot blue
            'marker': 's'
        },
        {
            'matmul_file': data_dir / 'matmul-t1.json',
            'salt_file': data_dir / 'matmul-t1-salt.json',
            'label': 'T1 (Tiled Once)',
            'color': '#ff7f0e',  # ggplot orange
            'marker': 'o'
        },
        {
            'matmul_file': data_dir / 'matmul-t2.json',
            'salt_file': data_dir / 'matmul-t2-salt.json',
            'label': 'T2 (Tiled Twice)',
            'color': '#2ca02c',  # ggplot green
            'marker': 'x'
        }
    ]
    
    # Create the plot
    plt.figure(figsize=(20, 7))
    
    # Additional tmp files corresponding to the groups (matmul, matmul-t1, matmul-t2)
    tmp_files = [data_dir / 'mr_t0.json', data_dir / 'mr_t1.json', data_dir / 'mr_t2.json']

    for i, group in enumerate(groups):
        # Load data from JSON files
        with open(group['matmul_file']) as f:
            matmul_data = json.load(f)
            
        with open(group['salt_file']) as f:
            salt_data = json.load(f)

        # Extract data for matmul
        matmul_blocks = np.array(matmul_data.get('blocks', []))
        matmul_miss_ratio = np.array(matmul_data.get('miss_ratio', []))

        # Merge any additional runs from tmp files and deduplicate overlapping points
        tmp_file = tmp_files[i] if i < len(tmp_files) else None
        if tmp_file:
            merged_blocks, merged_miss = merge_additional_runs(matmul_blocks, matmul_miss_ratio, tmp_file)
            matmul_blocks, matmul_miss_ratio = merged_blocks, merged_miss

        # Extract data for salt step function
        salt_turning_points = np.array(salt_data['miss_ratio_curve']['turning_points'])
        salt_miss_ratio = np.array(salt_data['miss_ratio_curve']['miss_ratio'])

        # For comparison metrics, interpolate the SALT predictions at matmul points
        salt_predictions = np.interp(matmul_blocks, salt_turning_points, salt_miss_ratio,
                                     left=1.0, right=0.0)

        # Calculate Mean Squared Error and MAPE (avoid divide-by-zero)
        if matmul_miss_ratio.size == 0:
            mse = float('nan')
            mape = float('nan')
            valid_mask = np.array([], dtype=bool)
        else:
            valid_mask = np.ones_like(matmul_miss_ratio, dtype=bool)
            mse = np.mean((matmul_miss_ratio[valid_mask] - salt_predictions[valid_mask]) ** 2)
            nonzero_mask = matmul_miss_ratio[valid_mask] != 0
            if nonzero_mask.any():
                mape = np.mean(np.abs((matmul_miss_ratio[valid_mask][nonzero_mask] - salt_predictions[valid_mask][nonzero_mask]) / matmul_miss_ratio[valid_mask][nonzero_mask])) * 100
            else:
                mape = float('nan')
        
        # print(f'{group["label"]} - MAPE: {mape:.2f}%, MSE: {mse:.2e}')

        # Plot cachegrind simulation
        if matmul_blocks.size > 0:
            marker = group.get('marker', 'x')
            # Make circle ('o') and square ('s') markers hollow by setting facecolors='none'.
            # For an unfilled marker like 'x', pass `color=` to avoid edgecolor warnings.
            if marker in ('o', 's'):
                plt.scatter(matmul_blocks[valid_mask], matmul_miss_ratio[valid_mask],
                            label=f'Cachegrind - {group["label"]}',
                            facecolors='none', edgecolors=group['color'], alpha=0.9,
                            s=300, marker=marker, linewidths=2.0)
            elif marker == 'x':
                plt.scatter(matmul_blocks[valid_mask], matmul_miss_ratio[valid_mask],
                            label=f'Cachegrind - {group["label"]}',
                            color=group['color'], alpha=0.9,
                            s=300, marker=marker, linewidths=2.0)
            else:
                plt.scatter(matmul_blocks[valid_mask], matmul_miss_ratio[valid_mask],
                            label=f'Cachegrind - {group["label"]}',
                            facecolors=group['color'], edgecolors=group['color'], alpha=0.8,
                            s=300, marker=marker, linewidths=2.0)

        # Plot salt step function using matplotlib's step function
        if len(salt_turning_points) > 0 and len(salt_miss_ratio) > 0:
            # Extend the last value to the right boundary of the plot to keep it as a plateau
            # Find the maximum x-value in the current plot to extend to
            plot_max_x = max(matmul_blocks.max(), salt_turning_points.max()) * 10
            
            # Add one more point at the end to extend the final value as a plateau
            # We add it at the last turning point and then at plot_max_x to keep it flat
            extended_turning_points = np.append(salt_turning_points, plot_max_x)
            extended_miss_ratio = np.append(salt_miss_ratio, salt_miss_ratio[-1])
            
            plt.step(extended_turning_points, extended_miss_ratio, 
                    where='post', label=f'SALT - {group["label"]}', 
                    color=group['color'], linestyle='--', linewidth=2.5, alpha=0.9)

    # Formatting with ggplot style
    plt.xlabel('Cache Size (#Blocks)', fontsize=30, fontweight='bold')
    plt.ylabel('Miss Ratio', fontsize=30, fontweight='bold')
    plt.legend(fontsize=18, frameon=True, fancybox=True, shadow=True, loc='lower left', markerscale=1.3)
    plt.xscale('log')
    plt.yscale('log')

    x_min = 1.5e0
    x_max = 3e4
    plt.xlim(left=x_min, right=x_max)
    # autoscale_visible_log_y(plt.gca(), x_min, x_max)

    # Increase tick label sizes for better readability
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)

    # Save and show plot
    plt.tight_layout()
    plt.savefig(svg_output, format='svg')
    plt.savefig(svg_output.replace('.svg', '.png'), format='png', dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark Graph Plotter - Multiple Groups Comparison')
    parser.add_argument('--svg-output', type=str, required=True, help='Output SVG file path')
    parser.add_argument(
        '--data-dir', type=Path, default=DEFAULT_DATA_DIR,
        help='directory containing Cachegrind and SALT JSON inputs',
    )

    args = parser.parse_args()
    
    run(args.svg_output, args.data_dir)


# python3 graph.py --svg-output tmp/merged.svg
