# -*- coding: utf-8 -*-
# @Time    : 2022/11/25 16:35
# @Author  : lv yuqian
# @File    : visualize_time.py
# @Email   : lvyuqian_email@163.com
import os
import numpy as np
from base_config import *
import pickle as pkl
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
from base_graph import BaseGraph
from collections import Counter
from typing import Union, List, Tuple, Dict


class TimeVisualize:
    def __init__(self):
        pass

    def time_visualize_all(self):
        plt.rc('font', family='Times New Roman')
        plt.figure(figsize=(18, 6))
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.axes().set_yscale('log')

        datasets = ['TVShow', 'LastFM', 'Facebook', 'DeezerEU',
                    'HepPh', 'AstroPh', 'CondMat',
                    'USAir', 'USPower',
                    'EDU', 'Indo', 'Arabic']
        methods = ['TNC', 'KNM', 'SV', 'ATNC']
        bar_colors = ['#001871', '#ff585d', '#ffb549', '#41b6e6']
        width = 0.25
        axis_gap = 1.5
        axis = np.arange(0, axis_gap * len(datasets), axis_gap)
        if len(methods) % 2 == 0:
            gaps = np.arange(-len(methods) // 2 + 1 / 2, len(methods) // 2, 1)
        else:
            gaps = np.arange(-int(len(methods) / 2), len(methods) // 2 + 1, 1)
        time_data = {method: [TIME_DATA[method][name] for name in datasets] for method in methods}
        for idx, method in enumerate(methods):
            plt.bar(axis + gaps[idx] * width,
                    height=time_data[method],
                    width=width,
                    color=bar_colors[idx],
                    alpha=0.9,
                    label=method.upper())
        plt.grid(axis='y', linestyle='dotted', color='#6b778d')
        plt.xticks(axis, datasets, rotation=0)
        plt.ylabel("Time Consumption (seconds)", size=26)
        plt.tick_params(labelsize=22)
        plt.legend(fontsize=22)

        plt.tight_layout()
        plt.savefig(f"./figures/time_all.pdf", format='pdf', dpi=1000)
        plt.show()
        pass

    def time_visualize_rest(self):
        plt.rc('font', family='Times New Roman')
        plt.figure(figsize=(10, 6))
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        plt.axes().set_yscale('log')

        names = ['Gowalla', 'Citeseer', 'RoadNet', 'Google']
        methods = ['SV', 'ATNC']
        bar_colors = ['#ffb549', '#41b6e6']
        width = 0.4
        axis_gap = 1.5
        axis = np.arange(0, axis_gap * len(names), axis_gap)
        if len(methods) % 2 == 0:
            gaps = np.arange(-len(methods) // 2 + 1 / 2, len(methods) // 2, 1)
        else:
            gaps = np.arange(-int(len(methods) / 2), len(methods) // 2 + 1, 1)
        time_data = {method: [TIME_DATA[method][name] for name in names] for method in methods}
        for idx, method in enumerate(methods):
            plt.bar(axis + gaps[idx] * width,
                    height=time_data[method],
                    width=width,
                    color=bar_colors[idx],
                    alpha=0.9,
                    label=method.upper())
        plt.grid(axis='y', linestyle='dotted', color='#6b778d')
        plt.xticks(axis, names, rotation=0)
        plt.tick_params(labelsize=40)
        plt.tight_layout()
        plt.savefig(f"./figures/time_rest.png", format='png', dpi=1000)
        plt.show()


if __name__ == "__main__":
    obj = TimeVisualize()
    obj.time_visualize_all()
    obj.time_visualize_rest()
