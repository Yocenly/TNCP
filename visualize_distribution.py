# -*- coding: utf-8 -*-
# @Time    : 2022/12/7 13:17
# @Author  : lv yuqian
# @File    : visualize_distribution.py
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


class VisualizeDistribution:
    def __init__(self, dataset: str):
        self.dataset = dataset

    @staticmethod
    def visualize_distribution():
        datasets = dataset_names
        methods = ['TNC', 'KNM']
        bar_colors = ['#001871', '#ff585d', '#ffb549', '#41b6e6']

        for dataset in datasets:
            plt.rc('font', family='Times New Roman')
            plt.figure(figsize=(10, 6))
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.axes().set_yscale('log')
            for idx, method in enumerate(methods):
                recorder = pkl.load(open(f"./recorders/{method}/{dataset}_recorder.pkl", "rb"))
                RC_list = [value['strength'] - value['stability']
                           for value in recorder.values() if value['strength'] != value['stability']]
                counter = Counter(RC_list)
                axis = np.arange(1, max(RC_list) + 1)
                data = [counter.get(x, 0) for x in axis]
                # print(len(np.arange(len(counter.keys()))), len(list(counter.values())))
                plt.bar(axis,
                        height=data,
                        color=bar_colors[idx],
                        alpha=0.7)
            plt.title(f"{dataset}")
            plt.xticks(axis, np.arange(1, max(RC_list) + 1), rotation=0)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    VisualizeDistribution.visualize_distribution()