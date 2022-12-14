# -*- coding: utf-8 -*-
# @Time    : 2022/11/25 15:59
# @Author  : lv yuqian
# @File    : method_tnc.py
# @Email   : lvyuqian_email@163.com
import os
import time
import numpy as np
import pickle as pkl
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from base_config import *
from base_graph import BaseGraph
from typing import Union, List, Tuple, Dict
from calculate_metrics import CalculateMetrics
from multiprocessing import Process, Manager, Pool


class MethodTNC(BaseGraph):
    def __init__(self, dataset: str):
        super(MethodTNC, self).__init__(pkl.load(open(f"./datasets/{dataset}.pkl", "rb")))
        self.dataset = dataset
        self.diff_nodes = []
        self.delta = 0
        self.method = "TNC"
        self.recorder = dict()

    def update_global_stability(self, root_node: int, graph: nx.Graph, k_core: Dict[int, int], verbose: bool = False):
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

        def filter_global_corona_pedigree(root: int, decrease_count: dict = None):
            influences = dict()
            decrease_count = decrease_count or dict()
            corona_nodes = set(self.extract_corona_nodes(graph, k_core[root], k_core, decrease_count))
            """
                1.对目标节点strength的影响最大;
                2.对目标节点的邻居影响最多;
                3.对目标节点的邻居影响幅度最大;
            """
            while len(corona_nodes) > 0:
                start_node = list(corona_nodes)[0]
                influenced_nodes, temp_count = dfs_influenced_area(start_node, decrease_count.copy())
                influenced_nbrs = influenced_nodes & set(nx.neighbors(graph, root_node))
                influences[start_node] = [
                    temp_count[root] - decrease_count.get(root, 0),
                    len(influenced_nbrs),
                    # sum([temp_count[nbr] - decrease_count.get(nbr, 0) for nbr in influenced_nbrs])
                ]
                corona_nodes = corona_nodes - influenced_nodes

            epicenter_node = self.sort_in_order(list(influences.keys()), influences, orders=[True] * 2)[0]
            return epicenter_node, influences

        decrease_count = {root_node: 0}
        graph.nodes[root_node].setdefault('stability', 0)
        graph.nodes[root_node].setdefault('path', [])
        time_cost = time.process_time()
        if graph.nodes[root_node]['strength'] == 1 or k_core[root_node] <= 1:
            graph.nodes[root_node]['stability'] = graph.nodes[root_node]['strength']
        else:
            support_nodes = self.find_support_nbrs(root_node, GENERAL_SUPPORT, graph)
            cached_edges = {root_node: []}
            deleted_edges = list()
            while True:
                epicenter, influences = filter_global_corona_pedigree(root_node, decrease_count.copy())
                if influences[epicenter][0] == 0:
                    candidate_nodes = [node for node in support_nodes
                                       if graph.nodes[node]['strength'] > decrease_count.get(node, 0)]
                    epicenter = min(candidate_nodes,
                                    key=lambda x: graph.nodes[x]['strength'] - decrease_count.get(x, 0))
                    decrease_count.setdefault(epicenter, 0)
                    decrease_count[epicenter] += 1
                    decrease_count[root_node] += 1
                    graph.nodes[root_node]['stability'] += 1
                    graph.nodes[root_node]['path'].append((epicenter, root_node))
                    support_nodes.remove(epicenter)
                    graph.remove_edges_from([(epicenter, root_node)])
                    deleted_edges.append((epicenter, root_node))
                else:
                    candidate_nodes = [nbr for nbr in nx.neighbors(graph, epicenter)
                                       if k_core[nbr] >= k_core[epicenter]
                                       and graph.nodes[nbr]['strength'] > decrease_count.get(nbr, 0)]
                    graph.nodes[root_node]['stability'] += 1
                    graph.nodes[root_node]['path'].append((epicenter, candidate_nodes[0]))
                    dfs_influenced_area(epicenter, decrease_count)
                if graph.nodes[root_node]['strength'] <= decrease_count[root_node]:
                    graph.add_edges_from(deleted_edges)
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
        if graph.nodes[root_node]['strength'] != graph.nodes[root_node]['stability']:
            self.diff_nodes.append(f"Node: {root_node}, {self.recorder[root_node]}")
            self.delta += graph.nodes[root_node]['strength'] - graph.nodes[root_node]['stability']

    def test_single_node(self, target_node: int):
        self.update_global_stability(target_node, self.graph, self.k_core, verbose=True)
        print(f"Node: {target_node}, {self.recorder[target_node]}")

    # def test_total_nodes(self, verbose=True):
    #     with tqdm(total=self.graph.number_of_nodes(), desc=self.name, ncols=150, colour='YELLOW') as bar:
    #         ticks = time.process_time()
    #         for node in self.graph.nodes:
    #             self.update_global_stability(node, self.graph, self.k_core, verbose=False)
    #             bar.set_postfix_str(
    #                 f"[{self.k_core[node]}, {self.data_recorder[node]['strength']}, {self.data_recorder[node]['stability']}];"
    #                 f" Puffy Counts: {len(self.diff_nodes)}; Total Delta: {self.strength_delta};"
    #                 f" Time Cost: {round(time.process_time() - ticks, 5)}")
    #             bar.update()
    #     # if verbose:
    #     #     for idx, item in enumerate(self.diff_nodes):
    #     #         print(f"({idx + 1}) {item}")
    #     pkl.dump(self.data_recorder, open(f"./figures/global/ours/{dataset}_recorder.pkl", "wb"))

    def test_total_nodes(self, nodes=None, verbose=True):
        target_nodes = nodes or self.graph.nodes
        with tqdm(total=len(target_nodes), desc=self.method, ncols=150, colour='YELLOW') as bar:
            ticks = time.process_time()
            for node in target_nodes:
                self.update_global_stability(node, self.graph, self.k_core, verbose=False)
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
        # dataset = DATASET_NAME
        edges = pkl.load(open(f"./datasets/{dataset}.pkl", "rb"))
        graph = nx.Graph(edges)
        obj = MethodTNC(dataset)
        print(f"Name: {dataset}, Nodes: {graph.number_of_nodes()}, "
              f"Edges: {graph.number_of_edges()}, MaxCore: {max(obj.k_core.values())}")
        obj.multiprocess_search(20)
        # for target in [5984]:
        #     obj.test_single_node(target)
        #     print(obj.graph.number_of_edges())
        #     obj.test_collapse(target, obj.recorder[target]['path'])
    # obj.test_single_node(47030)