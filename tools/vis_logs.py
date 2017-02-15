import os,sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse

def read_log_file_with(fn, keywords=[]):
    r = []
    with open(fn) as f:
        for line in f:
            found = True
            for t in keywords:
                if t not in line:
                    found = False
                    break
            if found:
                r.append(line)
    return r


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize the log file')
    parser.add_argument('--log_file', dest='log_file', 
            help='The path to log file',
            required=True)
    parser.add_argument('--vis_loss', dest='vis_loss', 
            help='Vis the loss along time line')

    args = parser.parse_args()

    if args.vis_loss:
        lines = read_log_file_with(args.log_file, ['Iteration', 'loss'])
        print lines