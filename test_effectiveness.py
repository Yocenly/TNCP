# -*- coding: utf-8 -*-
# @Time    : 2022/12/6 20:56
# @Author  : lv yuqian
# @File    : test_effectiveness.py
# @Email   : lvyuqian_email@163.com
from base_config import *
import pickle as pkl
import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
from base_graph import BaseGraph


if __name__ == "__main__":
    for dataset in dataset_names:
        # dataset = 'deezereu'
        edges = pkl.load(open(f"./datasets/{dataset}.pkl", "rb"))
        recorder = pkl.load(open(f"./recorders/TNC/{dataset}_recorder.pkl", "rb"))
        graph = nx.Graph(edges)
        obj = BaseGraph(graph)
        print(f"{dataset}: Total Node Num: {len(recorder)}")
        count = 0
        for node in tqdm(recorder.keys()):
            if recorder[node]['strength'] > recorder[node]['stability']:
                obj.test_result(node, recorder[node]['path'])
                count += 1
        print(count)
