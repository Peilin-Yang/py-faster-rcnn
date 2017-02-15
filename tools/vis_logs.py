import os,sys
import re
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

def parse_loss_lines(lines):
    r = [[], []]
    c = re.compile(r'Iteration (.*)?, loss = (.*)?')
    for line in lines:
        m = c.search(line)
        if m:
            r[0].append(int(m.group(1)))
            r[1].append(float(m.group(2)))
    return r

def vis_loss(data):
    x = data[0]
    y = data[1]
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y, 'b-', label='loss over iteration')
    ax.legend(loc='best', frameon=True)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize the log file')
    parser.add_argument('--log_file', dest='log_file', 
            help='The path to log file',
            required=True)
    parser.add_argument('--vis_loss', action='store_true', 
            help='Vis the loss along time line')

    args = parser.parse_args()

    if args.vis_loss:
        lines = read_log_file_with(args.log_file, ['Iteration', 'loss'])
        parsed = parse_loss_lines(lines)
        vis_loss(parsed)