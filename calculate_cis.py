# -*- coding: utf-8 -*-
# @Time    : 2022/11/25 15:55
# @Author  : lv yuqian
# @File    : calculate_cis.py
# @Email   : lvyuqian_email@163.com
import pickle as pkl
import networkx as nx
from tqdm import tqdm
from base_config import *
from base_graph import BaseGraph


class CalculateCIS(BaseGraph):
    def __init__(self, graph: nx.Graph):
        super(CalculateCIS, self).__init__(graph)
        self.weak_support = dict()
        self.candidate_set = dict()
        for node in tqdm(self.graph.nodes):
            self.weak_support[node] = self.find_support_nbrs(node, WEAK_SUPPORT, self.graph)
        for u in tqdm(self.graph.nodes):
            self.candidate_set[u] = [node for node in self.find_support_nbrs(u, NONE_SUPPORT, self.graph)
                                     if len(self.weak_support[node]) < self.k_core[node]]

    def init_influence(self, used_metric: str = "strength"):
        assert used_metric in ['strength', 'stability'], f"Illegal used_metric {used_metric} given."
        sorted_nodes = sorted(self.graph.nodes, key=lambda x: self.k_core[x])
        for u in sorted_nodes:
            candidate_set = self.candidate_set[u]
            influence_list = []
            for v in candidate_set:
                self.recorder[v].setdefault('influence', 1)
                delta = 1 - len(self.weak_support[v]) / self.k_core[v]
                strong_support = self.recorder[v]['strength'] + self.k_core[v] - 1 - len(self.weak_support[v])
                influence_list.append(delta * self.recorder[v]['influence'] / strong_support)
            self.recorder[u]['influence'] = max(sum(influence_list), 1)

    def calculate_cis(self, recorder: dict, used_metric: str = "strength", threshold: float = 0.95):
        self.recorder = recorder.copy()
        self.init_influence(used_metric)
        influence = [values['influence'] for values in self.recorder.values()]
        influence_threshold = sorted(influence)[int(threshold * len(self.recorder))]
        S_f = [node for node in self.graph.nodes if self.recorder[node]['influence'] >= influence_threshold]
        return sum([self.recorder[node][used_metric] for node in S_f]) / len(S_f)

if __name__ == "__main__":
    for dataset in dataset_names:
        graph = nx.Graph(pkl.load(open(f"./datasets/{dataset}.pkl", "rb")))
        recorder = pkl.load(open(f"./recorders/ATNC/{dataset}_recorder.pkl", "rb"))
        print(dataset)
        cis_obj = CalculateCIS(graph)
        cs_cis = cis_obj.calculate_cis(recorder)
        ns_cis = cis_obj.calculate_cis(recorder, 'stability')
        print(f"CS-based CIS: {cs_cis: .5f}; NS-based CIS: {ns_cis: .5f}.")