# -*- coding: utf-8 -*-
# @Time    : 2022/11/25 16:08
# @Author  : lv yuqian
# @File    : calculate_metrics.py
# @Email   : lvyuqian_email@163.com
import os
import numpy as np
import pickle as pkl
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from base_config import *
from base_graph import BaseGraph
from collections import Counter

class CalculateMetrics:
    def __init__(self, method: str, dataset: str):
        self.method = method
        self.dataset = dataset
        self.SCS = 0
        self.NBN = 0
        self.RC_list = list()
        self.recorder = pkl.load(open(f"./recorders/{method}/{dataset}_recorder.pkl", "rb"))

    def calculate_metrics(self):
        for node, value in self.recorder.items():
            self.SCS += value['strength']
            if value['strength'] != value['stability']:
                self.NBN += 1
                self.RC_list.append(value['strength'] - value['stability'])
        counter = dict(sorted(Counter(self.RC_list).items(), reverse=True))
        SRC = sum(self.RC_list)
        WAR = sum([key * (1 / value) for key, value in counter.items()]) / \
              sum([1 / value for value in counter.values()] + [10e-5])
        # WAR = sum([rc * 1 / counter[rc] for rc in self.RC_list]) / sum([1 / counter[rc] for rc in self.RC_list])
        # WAR = sum([key * value for key, value in counter.items()]) / len(counter)
        # WAR = sum(self.RC_list) / len(self.RC_list)
        # print(counter)
        print(f"{self.method.upper()}-{self.dataset} --> "
              f"NS: {self.NBN}; RC: {SRC: .2f}; WAR: {WAR: .4f}; RP: {SRC / self.SCS * 100: .6f}")


if __name__ == "__main__":
    for method in method_names:
        for dataset in dataset_names:
            if not os.path.exists(f"./recorders/{method}/{dataset}_recorder.pkl"):
                print(f"There is no file named {method}/{dataset}_recorder.pkl.")
                continue
            obj = CalculateMetrics(method, dataset)
            obj.calculate_metrics()
        print()