#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import argparse
import tempfile
import os
import json
import time
from concurrent.futures import ProcessPoolExecutor


def geometric_blocks(max_cache, growth_factor):
    """Return a sparse sweep combining geometric samples and powers of two."""
    blocks = {2, max_cache}

    value = 2
    while value < max_cache:
        blocks.add(value)
        value = max(value + 1, int(value * growth_factor))

    value = 2
    while value < max_cache:
        blocks.add(value)
        value *= 2

    return sorted(value for value in blocks if value <= max_cache)


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
    parser.add_argument('--max-cache', type=int, default=1024, help='maximum number of cache blocks')
    parser.add_argument('--block', type=int, default=64, help='block size')
    parser.add_argument('--step', type=int, default=1, help='step size')
    parser.add_argument(
        '--sampling', choices=('dense', 'geometric'), default='dense',
        help='generate a dense or sparse geometric sweep (default: dense)',
    )
    parser.add_argument(
        '--growth-factor', type=float, default=1.5,
        help='growth factor for geometric sampling (default: 1.5)',
    )
    parser.add_argument(
        '--blocks', type=str,
        help='comma-separated cache-block counts; overrides generated sampling options',
    )
    parser.add_argument('--output', type=str, default='/tmp/miss_ratio.json', help='output file')
    args = parser.parse_args()
    if args.max_cache < 2:
        parser.error('--max-cache must be at least 2')
    if args.step < 1:
        parser.error('--step must be at least 1')
    if args.growth_factor <= 1:
        parser.error('--growth-factor must be greater than 1')

    src = os.path.abspath(args.src)
    all_task_args = []
    if args.blocks:
        blocks_to_run = [int(value) for value in args.blocks.split(',')]
    elif args.sampling == 'geometric':
        blocks_to_run = geometric_blocks(args.max_cache, args.growth_factor)
    else:
        blocks_to_run = list(range(2, args.max_cache + 1, args.step))
    for blocks in blocks_to_run:
        all_task_args.append((src, blocks, args.block))
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(execute, all_task_args, chunksize=1))
    end_time = time.time()
    map = {}
    map['miss_ratio'] = results
    map['blocks'] = blocks_to_run
    map['time_elapsed'] = end_time - start_time
    with open(args.output, 'w') as f:
        json.dump(map, f, indent=4)

if __name__ == '__main__':
    main()
