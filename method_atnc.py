# -*- coding: utf-8 -*-
# @Time    : 2022/11/25 16:21
# @Author  : lv yuqian
# @File    : method_atnc.py
# @Email   : lvyuqian_email@163.com
import os
import pickle as pkl
import time

import numpy as np
from tqdm import tqdm
import networkx as nx
from base_config import *
from base_graph import BaseGraph
from typing import Union, List, Tuple, Dict
from calculate_metrics import CalculateMetrics
from multiprocessing import Process, Manager, Pool


class MethodATNC(BaseGraph):
    def __init__(self, dataset: str):
        super(MethodATNC, self).__init__(pkl.load(open(f"./datasets/{dataset}.pkl", "rb")))
        self.dataset = dataset
        self.diff_nodes = []
        self.delta = 0
        self.method = "ATNC"
        self.recorder = dict()
        self.search_count = 0

    def update_node_stability(self, root_node: int, graph: nx.Graph, k_core: Dict[int, int], verbose: bool = False):
        def dfs_influenced_area(root: int, decrease_count: dict = None, influenced_set: set = None):
            decrease_count, influenced_set = decrease_count or dict(), influenced_set or set()
            self.search_count += 1
            decrease_count.setdefault(root, 0)
            decrease_count[root] += 1
            root_delta = graph.nodes[root]['strength'] - decrease_count[root]
            influenced_set.add(root)
            for nbr in nx.neighbors(graph, root):
                nbr_delta = graph.nodes[nbr]['strength'] - decrease_count.get(nbr, 0)
                if k_core[root] == k_core[nbr] and root_delta < 1 and nbr_delta >= 1 and nbr != root_node:
                    dfs_influenced_area(nbr, decrease_count, influenced_set)
            return influenced_set, decrease_count

        def check_corona_node(root: int) -> list:
            influenced_nodes, temp_count = dfs_influenced_area(root, decrease_count.copy())
            influenced_nbrs = [nbr for nbr in nx.neighbors(graph, root_node) if nbr in influenced_nodes]
            """
                1. 使邻居节点坍塌数量最多;
                2. 受影响的邻居节点数量最多;
                3. 邻居节点的稳定程度下降幅度最大;
                4. 邻居节点剩余稳定程度最少;
                5. 全图影响节点的数量最多;
            """
            collapsed_nbrs = [nbr for nbr in influenced_nbrs if graph.nodes[nbr]['strength'] - temp_count[nbr] < 1]
            delta_strength = [temp_count[node] - decrease_count.get(node, 0) for node in influenced_nbrs]
            rest_strength = [graph.nodes[node]['strength'] - temp_count.get(node, 0) for node in influenced_nbrs]
            return [len(collapsed_nbrs),
                    len(influenced_nbrs),
                    # np.sum(delta_strength),
                    # np.sum(rest_strength),
                    # len(influenced_nodes),
                    ], collapsed_nbrs

        def filter_local_corona_pedigree(root: int) -> list:
            corona_nbrs = [nbr for nbr in nx.neighbors(graph, root_node) if k_core[root] == k_core[nbr]
                            and graph.nodes[nbr]['strength'] - decrease_count.get(nbr, 0) == 1]
            influences = dict()
            while corona_nbrs:
                influences[corona_nbrs[0]], collapsed_nbrs = check_corona_node(corona_nbrs[0])
                corona_nbrs = list(set(corona_nbrs) - set(collapsed_nbrs))
            try:
                return self.sort_in_order(list(influences.keys()), influences, orders=[True, True])# , True, False, True
            except:
                return corona_nbrs

        decrease_count = {root_node: 0}
        graph.nodes[root_node].setdefault('stability', 0)
        graph.nodes[root_node].setdefault('path', [])
        time_cost = time.process_time()
        corona_nodes = filter_local_corona_pedigree(root_node)
        if graph.nodes[root_node]['strength'] == 1 or len(corona_nodes) == 0 or k_core[root_node] <= 1:
            graph.nodes[root_node]['stability'] = graph.nodes[root_node]['strength']
        else:
            while len(corona_nodes) > 0:
                self.search_count = 0
                # print("Corona: ", len(corona_nodes), graph.nodes[root_node]['strength'])
                filtered_nodes, _ = dfs_influenced_area(corona_nodes[0], decrease_count)
                # print("Count: ", self.search_count)
                collapsed_nodes = [node for node in filtered_nodes if
                                   graph.nodes[node]['strength'] - decrease_count.get(node, 0) < 1]
                decrease_count[root_node] += len(set(collapsed_nodes) & set(nx.neighbors(graph, root_node)))
                graph.nodes[root_node]['stability'] += 1
                graph.nodes[root_node]['path'].append((corona_nodes[0], root_node))
                if verbose:
                    print(f"{root_node}, {corona_nodes}, {collapsed_nodes}, {decrease_count.get(root_node, 0)}")
                if graph.nodes[root_node]['strength'] - decrease_count.get(root_node, 0) < 1:
                    break
                corona_nodes = filter_local_corona_pedigree(root_node)
            if len(corona_nodes) == 0 and graph.nodes[root_node]['strength'] - decrease_count.get(root_node, 0) >= 1:
                delta = graph.nodes[root_node]['strength'] - decrease_count[root_node]
                graph.nodes[root_node]['stability'] += delta
                supported_nbrs = [nbr for nbr in nx.neighbors(graph, root_node) if k_core[root_node] <= k_core[nbr]
                                   and graph.nodes[nbr]['strength'] > decrease_count.get(nbr, 0)]
                graph.nodes[root_node]['path'].extend([(nbr, root_node) for nbr in supported_nbrs[:delta]])
        self.recorder[root_node] = {
            "core": k_core[root_node],
            "strength": graph.nodes[root_node]['strength'],
            "stability": graph.nodes[root_node]['stability'],
            "path": graph.nodes[root_node]['path'],
            "time": time.process_time() - time_cost,
        }
        if graph.nodes[root_node]['strength'] != graph.nodes[root_node]['stability']:
            self.diff_nodes.append(f"Node: {root_node}, {self.recorder[root_node]}")
            self.delta += graph.nodes[root_node]['strength'] - graph.nodes[root_node]['stability']

    def test_single_node(self, target_node: int):
        self.update_node_stability(target_node, self.graph, self.k_core, verbose=True)
        print(f"Node: {target_node}, {self.recorder[target_node]}")

    def test_total_nodes(self, nodes = None, verbose=True):
        target_nodes = nodes or self.graph.nodes
        with tqdm(total=len(target_nodes), desc=self.method, ncols=150, colour='YELLOW') as bar:
            ticks = time.process_time()
            for node in target_nodes:
                self.update_node_stability(node, self.graph, self.k_core, verbose=False)
                bar.set_postfix_str(
                    f"[{self.k_core[node]}, {self.recorder[node]['strength']}, {self.recorder[node]['stability']}];"
                    f" Puffy Counts: {len(self.diff_nodes)}; Total Delta: {self.delta};"
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
        pkl.dump(self.recorder, open(f"./recorders/{self.method}/{self.dataset}_recorder.pkl", "wb"))
        print(f"{dataset} --> Bubble Counts: {self.diff_num}; "
              f"Total Delta: {self.delta: .4f}; Time Cost: {self.time_cost: .4f}")
        obj = CalculateMetrics(self.method, self.dataset)
        obj.calculate_metrics()


if __name__ == "__main__":
    for dataset in dataset_names:
        edges = pkl.load(open(f"./datasets/{dataset}.pkl", "rb"))
        graph = nx.Graph(edges)
        obj = MethodATNC(dataset)
        print(f"Name: {dataset}, Nodes: {graph.number_of_nodes()}, "
              f"Edges: {graph.number_of_edges()}, MaxCore: {max(obj.k_core.values())}")
        obj.multiprocess_search(20)