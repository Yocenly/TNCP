# -*- coding: utf-8 -*-
# @Time    : 2022/11/25 16:22
# @Author  : lv yuqian
# @File    : method_knm.py
# @Email   : lvyuqian_email@163.com
import os
import pickle as pkl
import time
import numpy as np
from tqdm import tqdm
import networkx as nx
from base_config import *
import matplotlib.pyplot as plt
from base_graph import BaseGraph
from typing import Union, List, Tuple, Dict
from calculate_metrics import CalculateMetrics
from multiprocessing import Process, Manager, Pool


class MethodKNM(BaseGraph):
    def __init__(self, dataset: str):
        super(MethodKNM, self).__init__(pkl.load(open(f"./datasets/{dataset}.pkl", "rb")))
        self.dataset = dataset
        self.method = 'KNM'
        self.delta = 0
        self.diff_nodes = []
        self.recorder = dict()

    def knm_search(self, root_node: int, graph: nx.Graph, k_core: Dict[int, int], verbose: bool = False):
        def dfs_influenced_area(root: int, decrease_count: dict = None, influenced_set: set = None):
            decrease_count, influenced_set = decrease_count or dict(), influenced_set or set()
            decrease_count.setdefault(root, 0)
            decrease_count[root] += 1
            root_delta = graph.nodes[root]['strength'] - decrease_count.get(root, 0)
            influenced_set.add(root)
            for nbr in nx.neighbors(graph, root):
                nbr_delta = graph.nodes[nbr]['strength'] - decrease_count.get(nbr, 0)
                if k_core[root] == k_core[nbr] and root_delta < 1 and nbr_delta >= 1:
                    dfs_influenced_area(nbr, decrease_count, influenced_set)
            return influenced_set, decrease_count

        def sort_candidate_nodes(root: int, decrease_count: dict = None):
            influences = dict()
            decrease_count = decrease_count or dict()
            corona_nodes = set(self.extract_corona_nodes(graph, k_core[root], k_core, decrease_count))
            while len(corona_nodes) > 0:
                start_node = list(corona_nodes)[0]
                influenced_nodes, temp_count = dfs_influenced_area(start_node, decrease_count.copy())
                collapsed_nodes = [node for node in influenced_nodes
                                   if self.graph.nodes[node]['strength'] <= temp_count[node]]
                influences[start_node] = [len(collapsed_nodes)]
                corona_nodes = corona_nodes - influenced_nodes

            epicenter_node = self.sort_in_order(list(influences.keys()), influences, orders=[True])[0]
            return epicenter_node, influences

        time_cost = time.process_time()
        decrease_count = {root_node: 0}
        graph.nodes[root_node].setdefault('stability', 0)
        graph.nodes[root_node].setdefault('path', [])

        if graph.nodes[root_node]['strength'] <= 1 or k_core[root_node] <= 1:
            graph.nodes[root_node]['stability'] = graph.nodes[root_node]['strength']
        else:
            while True:
                epicenter, influences = sort_candidate_nodes(root_node, decrease_count.copy())
                supported_nodes = [nbr for nbr in nx.neighbors(graph, epicenter) if k_core[epicenter] <= k_core[nbr]
                                   and graph.nodes[nbr]['strength'] > decrease_count.get(nbr, 0)]
                graph.nodes[root_node]['stability'] += 1
                graph.nodes[root_node]['path'].append((epicenter, supported_nodes[0]))
                dfs_influenced_area(epicenter, decrease_count)
                # if verbose:
                #     print(epicenter, '\n', influences, '\n', decrease_count)
                if (graph.nodes[root_node]['strength'] <= decrease_count[root_node]
                    or graph.nodes[root_node]['strength'] <= graph.nodes[root_node]['stability']):
                    break
            # if graph.nodes[root_node]['strength'] > decrease_count.get(root_node, 0):
            #     delta = graph.nodes[root_node]['strength'] - decrease_count[root_node]
            #     graph.nodes[root_node]['stability'] += delta
            #     supported_nodes = [nbr for nbr in nx.neighbors(graph, root_node) if k_core[root_node] <= k_core[nbr]
            #                        and graph.nodes[nbr]['strength'] > decrease_count.get(nbr, 0)]
            #     graph.nodes[root_node]['path'].extend([(nbr, root_node) for nbr in supported_nodes[:delta]])
        self.recorder[root_node] = {
            "core": k_core[root_node],
            "strength": graph.nodes[root_node]['strength'],
            "stability": graph.nodes[root_node]['stability'],
            "path": graph.nodes[root_node]['path'],
            "time": time.process_time() - time_cost,
        }
        if self.recorder[root_node]['strength'] > self.recorder[root_node]['stability']:
            self.diff_nodes.append(f"Node: {root_node}, {self.recorder[root_node]}")
            self.delta += self.recorder[root_node]['strength'] - self.recorder[root_node]['stability']

    # def test_total_nodes(self, verbose=True):
    #     with tqdm(total=self.graph.number_of_nodes(), desc=self.name, ncols=150, colour='YELLOW') as bar:
    #         ticks = time.process_time()
    #         for node in self.graph.nodes:
    #             self.knm_search(node, self.graph, self.k_core, verbose=False)
    #             bar.set_postfix_str(
    #                 f"[{self.k_core[node]}, {self.recorder[node]['strength']}, {self.recorder[node]['stability']}];"
    #                 f" Puffy Counts: {len(self.diff_nodes)}; Total Delta: {self.delta};"
    #                 f" Time Cost: {round(time.process_time() - ticks, 5)}")
    #             bar.update()
    #     if verbose:
    #         for idx, item in enumerate(self.diff_nodes):
    #             print(f"({idx + 1}) {item}")
    #     pkl.dump(self.recorder, open(f"./figures/global/knm/{dataset}_recorder.pkl", "wb"))

    def test_total_nodes(self, nodes = None, verbose=True):
        target_nodes = nodes or self.graph.nodes
        with tqdm(total=len(target_nodes), desc=self.method, ncols=150, colour='YELLOW') as bar:
            ticks = time.process_time()
            for node in target_nodes:
                self.knm_search(node, self.graph, self.k_core, verbose=False)
                bar.set_postfix_str(
                    f"[{self.k_core[node]}, {self.recorder[node]['strength']}, {self.recorder[node]['stability']}];"
                    f" Puffy Counts: {len(self.diff_nodes)}; Total Delta: {self.delta};"
                    f" Time Cost: {round(time.process_time() - ticks, 4)}")
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
        pkl.dump(self.recorder, open(f"./recorders/{self.method}/{self.dataset}_recorder.pkl", "wb"))
        print(f"{dataset} --> Bubble Counts: {self.diff_num}; "
              f"Total Delta: {self.delta: .4f}; Time Cost: {self.time_cost: .4f}")
        obj = CalculateMetrics(self.method, self.dataset)
        obj.calculate_metrics()


if __name__ == "__main__":
    for dataset in dataset_names:
        # dataset = 'brightkite'
        edges = pkl.load(open(f"./datasets/{dataset}.pkl", "rb"))
        graph = nx.Graph(edges)
        obj = MethodKNM(dataset)
        print(f"Name: {dataset}, Nodes: {graph.number_of_nodes()}, "
              f"Edges: {graph.number_of_edges()}, MaxCore: {max(obj.k_core.values())}")
        obj.multiprocess_search(20)