#/usr/bin/env python
##
#  @author Rakesh Shrestha, rakeshs@sfu.ca

import json
import argparse
from collections import OrderedDict
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='compare results')
    parser.add_argument('results1_log', type=str)
    parser.add_argument('results2_log', type=str)
    return parser.parse_args()

def process_results(results):
    return {
        tuple(key.split('_')[:2]): value
        for key, value in results.items()
    }

if __name__ == '__main__':
    args = parse_args()
    with open(args.results1_log, 'r') as f:
        results1 = process_results(json.load(f))
    with open(args.results2_log, 'r') as f:
        results2 = process_results(json.load(f))

    tau_idx = 1
    results_diff = {
        key: {
            'diff': results1[key][tau_idx] - results2[key][tau_idx],
            'results1': results1[key][tau_idx],
            'results2': results2[key][tau_idx]
        }
        for key in results1.keys()
    }

    labels = list(set([i[0] for i in results1.keys()]))
    name = {'02828884': 'bench', '03001627': 'chair', '03636649': 'lamp', '03691459': 'speaker', '04090263': 'firearm',
            '04379243': 'table', '04530566': 'watercraft', '02691156': 'plane', '02933112': 'cabinet',
            '02958343': 'car', '03211117': 'monitor', '04256520': 'couch', '04401088': 'cellphone'}
    # group by labels
    grouped_diff = {i: {} for i in labels}
    for (label, label_appendix), diff in results_diff.items():
        grouped_diff[label][label_appendix] = diff

    sort_key = 'diff'
    num_show = 20
    sorted_diff = {
        '/'.join((name[label], label)): OrderedDict(
            sorted(
                diff.items(), key=lambda x: x[1][sort_key], reverse=True
            )[:num_show]
        )
        for label, diff in grouped_diff.items()
    }

    print(json.dumps(sorted_diff, indent=4))
