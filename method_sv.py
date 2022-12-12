# -*- coding: utf-8 -*-
# @Time    : 2022/11/25 16:22
# @Author  : lv yuqian
# @File    : method_sv.py
# @Email   : lvyuqian_email@163.com
import os
import time
import random
import itertools
import numpy as np
import pickle as pkl
import networkx as nx
from tqdm import tqdm
from base_config import *
from base_graph import BaseGraph
from typing import Union, List, Tuple, Dict
from calculate_metrics import CalculateMetrics
from multiprocessing import Process, Manager, Pool


class MethodSV(BaseGraph):
    def __init__(self, dataset: str):
        super(MethodSV, self).__init__(pkl.load(open(f"./datasets/{dataset}.pkl", "rb")))
        self.dataset = dataset
        self.init_shapley_layers()
        self.method = "SV"
        self.delta = 0
        self.diff_nodes = []
        self.recorder = dict()

    def cal_layer_gain(self, target_k):
        edge_gain = dict()
        k_graph = nx.Graph(nx.k_core(self.graph, target_k))
        higher_core = nx.Graph(nx.k_core(self.graph, target_k + 1))
        k_graph.remove_edges_from(list(higher_core.edges()))
        candidate_edges = k_graph.edges()
        # print(len(candidate_edges))
        sample_times = int(round(np.log(k_graph.number_of_edges()) / 0.1))
        for _ in range(sample_times):
            temp_graph = k_graph.copy()
            k_core = self.k_core.copy()
            shuffled_edges = random.sample(candidate_edges, len(candidate_edges))
            for edge in shuffled_edges:
                edge_gain.setdefault(edge, 0)
                edge_gain[edge] += len(self.update_k_core_after_deletion(temp_graph, [edge], k_core))
        self.layer_gain[target_k] = sorted(list(edge_gain.keys()), key=lambda x: edge_gain[x], reverse=True)

    def init_shapley_layers(self):
        self.layer_gain = dict()
        for i in tqdm(set(self.k_core.values()) - set([1])):
            if i == 1:
                continue
            self.cal_layer_gain(i)

    def sv_search(self, target_node: int):
        time_cost = time.process_time()
        self.graph.nodes[target_node].setdefault('stability', self.graph.nodes[target_node]['strength'])
        self.graph.nodes[target_node].setdefault('path', [])
        if not (self.k_core[target_node] <= 1 or self.graph.nodes[target_node]['strength'] <= 1):
            sv_stability = list()
            removed_edges = list()
            k_core = self.k_core.copy()
            while True:
                removed_edges.extend([self.layer_gain[self.k_core[target_node]][len(removed_edges)]])
                modified_nodes = self.update_k_core_after_deletion(self.graph, removed_edges, k_core)
                if target_node in modified_nodes or len(removed_edges) >= self.graph.nodes[target_node]['strength']:
                    break
            sv_stability.append(len(removed_edges))
            self.graph.add_edges_from(removed_edges)
            self.graph.nodes[target_node]['stability'] = np.mean(sv_stability)
            self.graph.nodes[target_node]['path'] = removed_edges
        self.recorder[target_node] = {
            "core": self.k_core[target_node],
            "strength": self.graph.nodes[target_node]['strength'],
            "stability": self.graph.nodes[target_node]['stability'],
            "path": self.graph.nodes[target_node]['path'],
            "time": time.process_time() - time_cost,
        }
        if self.recorder[target_node]['strength'] > self.recorder[target_node]['stability']:
            self.diff_nodes.append(f"Node: {target_node}, {self.recorder[target_node]}")
            self.delta += self.recorder[target_node]['strength'] - self.recorder[target_node]['stability']

    def test_total_nodes(self, nodes=None, verbose=True):
        target_nodes = nodes or self.graph.nodes
        with tqdm(total=len(target_nodes), desc=self.method, ncols=150, colour='YELLOW') as bar:
            ticks = time.process_time()
            for node in target_nodes:
                self.sv_search(node)
                bar.set_postfix_str(
                    f"[{self.k_core[node]}, {self.recorder[node]['strength']}, {self.recorder[node]['stability']}];"
                    f" Puffy Counts: {len(self.diff_nodes)}; Total Delta: {self.delta: .4f};"
                    f" Time Cost: {round(time.process_time() - ticks, 4)}")
                bar.update()
        return self.recorder, len(self.diff_nodes), self.delta, time.process_time() - ticks
        # if verbose:
        #     for idx, item in enumerate(self.diff_nodes):
        #         print(f"({idx + 1}) {item}")
        # pkl.dump(self.data_recorder, open(f"./visualization/local/ours/{dataset}_recorder.pkl", "wb"))

    def multiprocess_search(self, num_workers: int = 1):
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
    for dataset in ['CondMat']:
        edges = pkl.load(open(f"./datasets/{dataset}.pkl", "rb"))
        graph = nx.Graph(edges)
        # for i in range(1, 20):
        #     core1 = nx.k_core(graph, i)
        #     print(core1.number_of_edges())
        # break
        cost = time.process_time()
        obj = MethodSV(dataset)
        print(f"Name: {dataset}, Nodes: {graph.number_of_nodes()}, "
              f"Edges: {graph.number_of_edges()}, MaxCore: {max(obj.k_core.values())}")
        cost = time.process_time() - cost
        print(f"Traver Time: {cost}")
        obj.multiprocess_search(20)
        # obj.init_shapley_layers()