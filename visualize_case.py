# -*- coding: utf-8 -*-
# @Time    : 2023/2/22 13:58
# @Author  : lv yuqian
# @File    : visualize_case.py
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

class CaseVisualize(BaseGraph):
    def __init__(self, dataset: str):
        edges = pkl.load(open(f"./datasets/{dataset}.pkl", "rb"))
        self.recorders = {
            "red": pkl.load(open(f"./recorders/RED/{dataset}_recorder.pkl", "rb")),
            "rnd": pkl.load(open(f"./recorders/RND/{dataset}_recorder.pkl", "rb")),
            "knm": pkl.load(open(f"./recorders/KNM/{dataset}_recorder.pkl", "rb")),
            "sv": pkl.load(open(f"./recorders/SV/{dataset}_recorder.pkl", "rb")),
            "tnc": pkl.load(open(f"./recorders/TNC/{dataset}_recorder.pkl", "rb")),
            "atnc": pkl.load(open(f"./recorders/ATNC/{dataset}_recorder.pkl", "rb")),
        }
        self.dataset = dataset
        super(CaseVisualize, self).__init__(edges)
        self.CaseVisualize = sorted(list(self.graph.nodes), key=lambda x: self.graph.nodes[x]['strength'], reverse=True)

    def get_trend_data(self, target_node, recorder):
        k_core = self.k_core.copy()
        graph = self.graph.copy()
        sn_trend = [len([nbr for nbr in nx.neighbors(graph, target_node) if k_core[nbr] >= self.k_core[target_node]])]
        for edge in recorder[target_node]['path']:
            self.update_k_core_after_deletion(graph, [edge], k_core)
            sn_trend.append(len([nbr for nbr in nx.neighbors(graph, target_node)
                                 if k_core[nbr] >= self.k_core[target_node]]))
        return sn_trend

    def draw(self, target_node: int):
        plt.rc('font', family='Times New Roman')
        # plt.figure(figsize=(6, 6))
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        trend_prc = self.get_trend_data(target_node, self.recorders['tnc'])
        trend_knm = self.get_trend_data(target_node, self.recorders['knm'])
        trend_sv = self.get_trend_data(target_node, self.recorders['sv'])
        trend_aprc = self.get_trend_data(target_node, self.recorders['atnc'])
        plt.plot(trend_prc, marker='o', markersize=11, markerfacecolor='white', label="TNC", color='#001871', linewidth=2)
        plt.plot(trend_knm, marker='s', markersize=10, markerfacecolor='white', label="KNM", color='#ff585d', linewidth=2)
        plt.plot(trend_sv, marker='x', markersize=9, markerfacecolor='white', label="SV", color='#ffb549', linewidth=2)
        plt.plot(trend_aprc, marker='v', markersize=10, markerfacecolor='white', label="ATNC", color='#41b6e6', linewidth=2)

        plt.plot(np.ones(max(len(trend_knm), len(trend_sv))) * self.recorders['tnc'][target_node]['core'],
                 color="red", linestyle='dotted')
        plt.title(
                  f" k={self.recorders['tnc'][target_node]['core']},"
                  f" CS={self.recorders['tnc'][target_node]['strength']},"
                  f" NS={self.recorders['tnc'][target_node]['stability']}", size=24)
        plt.xlabel("Number of Deleted Edges", size=26)
        plt.ylabel("Number of Support Neighbors", size=24)
        plt.tick_params(labelsize=24)
        # plt.legend(fontsize=18, loc=(0.02, 0.02))
        plt.tight_layout()
        plt.savefig(f"./figures/{self.dataset}_{target_node}.pdf", format='pdf', dpi=1000)
        plt.show()


if __name__ == "__main__":
    # obj = TimeVisualize()
    # obj.time_visualize_all()
    # obj.time_visualize_rest()
    # obj = TrendVisualize('deezereu')
    # obj.part_traverse_visualize()
    obj = CaseVisualize('DeezerEU')
    obj.draw(365)