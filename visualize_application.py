# -*- coding: utf-8 -*-
# @Time    : 2022/11/23 21:23
# @Author  : lv yuqian
# @File    : visualize_application.py
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


class ApplicationVisualize:
    def __init__(self):
        pass

    @staticmethod
    def visualize_application():
        plt.rc('font', family='Times New Roman')
        plt.figure(figsize=(18, 6))
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        # plt.axes().set_ylim(0, 10 ** 5)
        # plt.axes().set_yscale('log')

        datasets = ['TVShow', 'LastFM', 'Facebook', 'DeezerEU', 'Gowalla',
                    'HepPh', 'AstroPh', 'CondMat', 'Citeseer',
                    'USAir', 'USPower', 'RoadNet',
                    'EDU', 'Indo', 'Arabic', 'Google']
        methods = ['ORI', 'ATNC']
        labels = ['CS-based', 'NS-based']
        bar_colors = ['#13334c', '#fd5f00']
        width = 0.25
        axis_gap = 1.5
        axis = np.arange(0, axis_gap * len(datasets), axis_gap)
        if len(methods) % 2 == 0:
            gaps = np.arange(-len(methods) // 2 + 1 / 2, len(methods) // 2, 1)
        else:
            gaps = np.arange(-int(len(methods) / 2), len(methods) // 2 + 1, 1)
        time_data = {method: [CIS_DATA[method][name] for name in datasets] for method in methods}
        for idx, method in enumerate(methods):
            plt.bar(axis + gaps[idx] * width,
                    height=time_data[method],
                    width=width,
                    color=bar_colors[idx],
                    alpha=0.9,
                    label=labels[idx])
        plt.grid(axis='y', linestyle='dotted', color='#6b778d')
        plt.xticks(axis, datasets, rotation=18)
        plt.yticks(range(0, 10, 2))
        plt.ylabel("CIS", size=26)
        plt.tick_params(labelsize=24)
        plt.legend(fontsize=26)

        plt.tight_layout(pad=0.2)
        plt.savefig(f"./figures/application.pdf", format='pdf', dpi=2000)
        plt.show()


if __name__ == '__main__':
    ApplicationVisualize.visualize_application()
