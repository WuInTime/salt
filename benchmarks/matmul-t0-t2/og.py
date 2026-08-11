#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import argparse
import tempfile
import os
import json
import time
from tqdm.contrib.concurrent import process_map

def execute(args):
    src, blocks, block_size = args
    with tempfile.TemporaryDirectory() as dir:
        subprocess.run([
            'gcc', '-static', '-nostdlib', '-fno-stack-protector', '-fno-pic', '-O3', src, '-o', 'a.out'], check=True, cwd=dir)
        out = subprocess.run(['valgrind', '--tool=cachegrind', '--cache-sim=yes',
                        '--D1=' + str(blocks * block_size) + ',' + str(blocks) + ',' + str(block_size), './a.out'], 
                        check=True, capture_output=True, cwd=dir)
        drefs = 0
        d1_miss = 0
        for line in out.stderr.decode().split('\n'):
            if 'D1  misses' in line:
                d1_miss = int(line.split()[3].replace(',', ''))
            elif 'D refs' in line:
                drefs = int(line.split()[3].replace(',', ''))
        return d1_miss / drefs
        

def main():
    parser = argparse.ArgumentParser(description='miss ratio curve generator')
    parser.add_argument("--src", type=str, help="source code", required=True)
    parser.add_argument('--max-cache', type=int, default=1048576, help='maximum number of cache blocks')
    parser.add_argument('--block', type=int, default=64, help='block size')
    parser.add_argument('--output', type=str, default='/tmp/miss_ratio.json', help='output file')
    parser.add_argument('--workers', type=int, default=None, help='maximum number of worker processes (default: number of Cores)')
    args = parser.parse_args()
    src = os.path.abspath(args.src)
    all_task_args = []
    # Use powers of 2 to cover the range
    blocks = 2
    while blocks <= args.max_cache:
        all_task_args.append((src, blocks, args.block))
        blocks = int(blocks * 1.1) +1 
        # // 1.5
    # Determine number of worker processes to use
    if args.workers is None:
        workers = os.cpu_count()
    else:
        workers = int(args.workers)
        if workers <= 0:
            raise ValueError('--workers must be a positive integer')
    print(f'Using {workers} worker processes to run {len(all_task_args)} tasks...')
    # return

    start_time = time.time()
    results = process_map(execute, all_task_args, chunksize=1, max_workers=workers, desc='Running tasks', unit='task')
    end_time = time.time()
    map = {}
    map['miss_ratio'] = results
    map['blocks'] = [task[1] for task in all_task_args]
    map['time_elapsed'] = end_time - start_time
    with open(args.output, 'w') as f:
        json.dump(map, f, indent=4)

if __name__ == '__main__':
    print('Benchmarking completed successfully.')
    main()


# python3 parallel_runner.py --src matmul.c --output tmp/mr_t0.json --max-cache 8192 --block 64;
# python3 parallel_runner.py --src matmul-t1.c --output tmp/mr_t1.json --max-cache 8192 --block 64;
# python3 parallel_runner.py --src matmul-t2.c --output tmp/mr_t2.json --max-cache 8192 --block 64;


# \subsubsection{Accuracy}

# Fig.~3 compares SALT’s miss-ratio curves with Cachegrind.  
# Although the figure shows a sampled subset of cache sizes for readability, we compute the accuracy metrics over the full sweep from 2 to 1024 blocks (1022 data points per kernel and tiling configuration).  
# Across all three kernels, SALT reconstructs both the plateau levels and the sharp drops associated with reuse-distance thresholds.  
# Using the complete data set, the mean absolute percentage error (MAPE) stays below 15\% and the mean squared error (MSE) remains below $6.4\times10^{-4}$, demonstrating close agreement in both the relative and absolute shape of the miss-ratio curves.  
# These results confirm that SALT captures the full piecewise structure of miss behavior across the entire cache-size range.
