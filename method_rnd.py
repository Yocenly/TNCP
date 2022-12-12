# -*- coding: utf-8 -*-
# @Time    : 2022/11/25 16:22
# @Author  : lv yuqian
# @File    : method_rnd.py
# @Email   : lvyuqian_email@163.com
import os
import random
import time
import numpy as np
import pickle as pkl
import networkx as nx
from tqdm import tqdm
from base_config import *
import matplotlib.pyplot as plt
from base_graph import BaseGraph
from typing import Union, List, Tuple
from calculate_metrics import CalculateMetrics
from multiprocessing import Process, Manager, Pool


class MethodRND(BaseGraph):
    def __init__(self, dataset: str):
        super(MethodRND, self).__init__(pkl.load(open(f"./datasets/{dataset}.pkl", "rb")))
        self.dataset = dataset
        self.method = "RND"
        self.delta = 0
        self.diff_nodes = []
        self.recorder = dict()

    def local_random_search(self, target_node: int, repeat_num: int = 10):
        time_cost = time.process_time()
        self.graph.nodes[target_node].setdefault('stability', self.graph.nodes[target_node]['strength'])
        if not (self.k_core[target_node] <= 1 or self.graph.nodes[target_node]['strength'] <= 1):
            random_stability = list()
            for epoch in range(repeat_num):
                removed_edges = list()
                k_core = self.k_core.copy()
                while True:
                    # support_nbrs = self.find_support_nodes(self.graph, target_node, GENERAL_SUPPORT)
                    # removed_edges.extend([(target_node, random.sample(support_nbrs, 1)[0])])
                    removed_edges.extend(random.sample(list(self.graph.edges(target_node)), 1))
                    modified_nodes = self.update_k_core_after_deletion(self.graph, removed_edges, k_core)
                    if target_node in modified_nodes or len(removed_edges) == self.graph.nodes[target_node]['strength']:
                        break
                random_stability.append(len(removed_edges))
                self.graph.add_edges_from(removed_edges)
            self.graph.nodes[target_node]['stability'] = np.mean(random_stability)
        self.recorder[target_node] = {
            "core": self.k_core[target_node],
            "strength": self.graph.nodes[target_node]['strength'],
            "stability": self.graph.nodes[target_node]['stability'],
            "time": time.process_time() - time_cost,
        }
        if self.recorder[target_node]['strength'] > self.recorder[target_node]['stability']:
            self.diff_nodes.append(f"Node: {target_node}, {self.recorder[target_node]}")
            self.delta += self.recorder[target_node]['strength'] - self.recorder[target_node]['stability']

    # def test_total_nodes(self, verbose=True):
    #     with tqdm(total=self.graph.number_of_nodes(), desc=self.name, ncols=150, colour='YELLOW') as bar:
    #         ticks = time.process_time()
    #         for node in self.graph.nodes:
    #             self.local_random_search(node, 10)
    #             bar.set_postfix_str(
    #                 f"[{self.k_core[node]}, {self.recorder[node]['strength']}, {self.recorder[node]['stability']}];"
    #                 f" Puffy Counts: {len(self.diff_nodes)}; Total Delta: {self.delta: .4f};"
    #                 f" Time Cost: {round(time.process_time() - ticks, 5)}")
    #             bar.update()
    #     if verbose:
    #         for idx, item in enumerate(self.diff_nodes):
    #             print(f"({idx + 1}) {item}")
    #     pkl.dump(self.recorder, open(f"./figures/local/random/{dataset}_recorder.pkl", "wb"))

    def test_total_nodes(self, nodes = None, verbose=True):
        target_nodes = nodes or self.graph.nodes
        with tqdm(total=len(target_nodes), desc=self.method, ncols=150, colour='YELLOW') as bar:
            ticks = time.process_time()
            for node in target_nodes:
                self.local_random_search(node, 10)
                bar.set_postfix_str(
                    f"[{self.k_core[node]}, {self.recorder[node]['strength']}, {self.recorder[node]['stability']}];"
                    f" Puffy Counts: {len(self.diff_nodes)}; Total Delta: {self.delta: .4f};"
                    f" Time Cost: {round(time.process_time() - ticks, 5)}")
                bar.update()
        return self.recorder, len(self.diff_nodes), self.delta, time.process_time() - ticks
        # if verbose:
        #     for idx, item in enumerate(self.diff_nodes):
        #         print(f"({idx + 1}) {item}")
        # pkl.dump(self.data_recorder, open(f"./figures/local/ours/{dataset}_recorder.pkl", "wb"))

    def multiprocess_search(self, num_workers: int = 1):
        cost = time.process_time()
        self.diff_num = 0
        self.time_cost = 0
        results = list()
        pool = Pool(processes=num_workers)
        gap = int(self.graph.number_of_nodes() / num_workers)
        nodes = list(self.graph.nodes)
        np.random.shuffle(nodes)
        for idx in range(num_workers - 1):
            results.append(pool.apply_async(func=self.test_total_nodes, args=([nodes[idx * gap: (idx + 1) * gap]])))
        results.append(pool.apply_async(func=self.test_total_nodes, args=([nodes[(num_workers - 1) * gap:]])))
        pool.close()
        pool.join()
        for result in results:
            self.recorder.update(result.get()[0])
            self.diff_num += result.get()[1]
            self.delta += result.get()[2]
            self.time_cost += result.get()[-1]
        print(f"{dataset} --> Bubble Counts: {self.diff_num}; "
              f"Total Delta: {self.delta: .4f}; Time Cost: {self.time_cost: .4f}")
        pkl.dump(self.recorder, open(f"./recorders/{self.method}/{self.dataset}_recorder.pkl", "wb"))
        obj = CalculateMetrics(self.method, self.dataset)
        obj.calculate_metrics()

if __name__ == "__main__":
    for dataset in dataset_names:
        # dataset = 'brightkite'
        # if os.path.exists(f"./figures/local/random/{dataset}_recorder.pkl"):
        #     continue
        edges = pkl.load(open(f"./datasets/{dataset}.pkl", "rb"))
        graph = nx.Graph(edges)
        obj = MethodRND(dataset)
        print(f"Name: {dataset}, Nodes: {graph.number_of_nodes()}, "
              f"Edges: {graph.number_of_edges()}, MaxCore: {max(obj.k_core.values())}")
        obj.multiprocess_search(20)