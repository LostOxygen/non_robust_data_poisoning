"""main file to plot attack results from previous attacks"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import argparse
from evaluation_methods.check_utils import plot_attack_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", "-p", help="Plot Prefix", type=str, required=True)
    args = parser.parse_args()
    plot_attack_results(**vars(args))
