# -*- coding: utf-8 -*-
# @Time    : 2022/11/25 15:55
# @Author  : lv yuqian
# @File    : base_graph.py
# @Email   : lvyuqian_email@163.com
import numpy as np
from base_config import *
import networkx as nx
from typing import Union, List, NoReturn, Dict


class Stack:
    def __init__(self, stack: list = None):
        self.stack = stack or list()

    def pop(self, index=-1):
        return self.stack.pop(index)

    def push(self, item):
        self.stack.extend(item)

    def is_empty(self):
        return True if len(self.stack) <= 0 else False

    def get_item(self, index):
        return self.stack[index] if index >= 0 and index < len(self.stack) else None

    def is_in(self, item):
        return True if item in self.stack else False

    def clear(self):
        temp, self.stack = self.stack.copy(), list()
        return temp


class Queue:
    def __init__(self, queue: list = None):
        self.queue = queue or list()

    def pop(self, index=0):
        return self.queue.pop(index)

    def push(self, item):
        self.queue.extend(item)

    def is_empty(self):
        return True if len(self.queue) <= 0 else False

    def get_item(self, index):
        return self.queue[index] if index >= 0 and index < len(self.queue) else None

    def is_in(self, item):
        return True if item in self.queue else False

    def clear(self):
        temp, self.queue = self.queue.copy(), list()
        return temp


class BaseGraph:
    def __init__(self, graph: Union[nx.Graph, List[tuple]]):
        self.graph = nx.Graph(graph)
        self.k_core = self.k_core_decomposition(self.graph)
        self.init_node_strength()

    @staticmethod
    def k_core_decomposition(graph: Union[nx.Graph, List[tuple]]) -> Dict[int, int]:
        """
        K核分解算法
        :param graph:
        :return:
        """
        # graph = graph.copy() if isinstance(graph, nx.Graph) else nx.Graph(graph)
        # k_core, k = dict(nx.degree(graph)), min(dict(nx.degree(graph)).values())
        # while True:
        #     while min(nx.degree(graph), key=lambda x: x[1])[1] <= k:
        #         for node in list(graph.nodes):
        #             if nx.degree(graph, node) <= k:
        #                 k_core[node] = k
        #                 graph.remove_node(node)
        #         if not graph:
        #             return k_core
        #     k = min(nx.degree(graph), key=lambda x: x[1])[1]
        return nx.core_number(graph)

    @staticmethod
    def update_k_core_after_deletion(graph: nx.Graph, edges: list, k_core: dict, value_change: bool = True):
        """
        删除一批连边后对k_core进行更新
        :param graph:
        :param edges:
        :param k_core:
        :return:
        """

        def update_edge_deletion(graph: nx.Graph, target_nodes: list, k_core: dict):
            """
            删除一条连边连边后对k_core进行更新
            :param graph:
            :param target_nodes:
            :param k_core:
            :return:
            """
            nodes_queue = Queue(target_nodes)
            modified_nodes = list()
            while not nodes_queue.is_empty():
                target_node = nodes_queue.pop()
                operative_neighbors = [nbr for nbr in nx.neighbors(graph, target_node) if k_core[nbr] >= k_core[target_node]]
                if len(operative_neighbors) >= k_core[target_node]:
                    continue
                else:
                    nodes_queue.push([nbr for nbr in operative_neighbors if k_core[nbr] <= k_core[target_node]
                                      and not (nbr in nodes_queue.queue)])
                    k_core[target_node] -= 1
                    modified_nodes.append(target_node)
            return modified_nodes

        modified_nodes = list()
        for edge in edges:
            graph.remove_edges_from([edge])
            if k_core[edge[0]] > k_core[edge[1]]:
                modified_nodes.extend(update_edge_deletion(graph, [edge[1]], k_core))
            elif k_core[edge[0]] < k_core[edge[1]]:
                modified_nodes.extend(update_edge_deletion(graph, [edge[0]], k_core))
            else:
                modified_nodes.extend(update_edge_deletion(graph, list(edge), k_core))
        if value_change is False:
            graph.add_edges_from(edges)
            for node in modified_nodes:
                k_core[node] += 1
        return modified_nodes


    def init_node_strength(self):
        """
        初始化节点核强度
        :return:
        """
        for node in self.graph.nodes:
            self.graph.nodes[node]['strength'] = self.cal_node_strength(node)

    def cal_node_strength(self, target_node: int, graph: nx.Graph = None, k_core: Dict[int, int] = None) -> int:
        """
        计算目标节点的核稳定程度, 即删除至少n+1条连边就可以使目标节点掉出当前核
        :param graph:
        :param target_node:
        :return:
        """
        used_graph = graph or self.graph
        used_core = k_core or self.k_core
        strength = len(self.find_support_nbrs(target_node, GENERAL_SUPPORT, used_graph))
        return strength - used_core[target_node] + 1

    def find_support_nbrs(self, target_node: int, relation: str = None, graph: nx.Graph = None, k_core: dict = None):
        assert relation in [NONE_SUPPORT, WEAK_SUPPORT, STRONG_SUPPORT, GENERAL_SUPPORT], f"Illegal param {relation}"
        used_graph = graph or self.graph
        used_core = k_core or self.k_core
        if relation == NONE_SUPPORT:
            return [nbr for nbr in nx.neighbors(used_graph, target_node) if used_core[nbr] < used_core[target_node]]
        elif relation == WEAK_SUPPORT:
            return [nbr for nbr in nx.neighbors(used_graph, target_node) if used_core[nbr] == used_core[target_node]]
        elif relation == STRONG_SUPPORT:
            return [nbr for nbr in nx.neighbors(used_graph, target_node) if used_core[nbr] > used_core[target_node]]
        elif relation == GENERAL_SUPPORT:
            return [nbr for nbr in nx.neighbors(used_graph, target_node) if used_core[nbr] >= used_core[target_node]]

    def extract_corona_nodes(self, graph: nx.Graph, target_k: int, k_core: dict = None, decrease_count: dict = None):
        k_core = k_core or self.k_core_decomposition(graph)
        decrease_count = decrease_count or dict()
        corona_nodes = [node for node in graph.nodes if k_core[node] == target_k and
                        graph.nodes[node].get('strength', self.cal_node_strength(node, graph, k_core))
                        - decrease_count.get(node, 0) == 1]
        return corona_nodes

    def collapse_validation(self, target_node: int, path: list):
        """
        验证各个算法连边删除对目标节点坍塌的有效性
        :param target_node:
        :param path:
        :return:
        """
        k_core = self.k_core.copy()
        graph = self.graph.copy()
        self.update_k_core_after_deletion(graph, path, k_core)
        if self.k_core[target_node] == k_core[target_node]:
            print(f"Attack Failed on Node {target_node} with Path {path}")

    def sort_in_order(self, array: list, elements: dict, orders: list) -> list:
        """
        多元排序, 按照elements中的元素数值进行排序, 顺序从后往前
        :param array:
        :param elements:
        :param orders:
        :return:
        """
        sorted_array = array.copy()
        for idx in range(len(elements[array[0]]) - 1, -1, -1):
            sorted_array = sorted(sorted_array, key=lambda x: elements[x][idx], reverse=orders[idx])
        return sorted_array

    def test_result(self, target_node, deleted_edges):
        graph, k_core = self.graph.copy(), self.k_core.copy()
        ori = self.k_core[target_node]
        # self.update_k_core_after_deletion(graph, deleted_edges, k_core)
        graph.remove_edges_from(deleted_edges)
        k_core = self.k_core_decomposition(graph)
        adv = k_core[target_node]
        if adv == ori:
            print(f"Failed of Node {target_node}, Core {ori}.")


if __name__ == "__main__":
    import pickle as pkl
    from collections import Counter

    dataset = "gowalla"
    edges = pkl.load(open(f"./datasets/{dataset}.pkl", "rb"))
    recorder = pkl.load(open(f"./results/global/prc/{dataset}_recorder.pkl", "rb"))
    graph = nx.Graph(edges)
    print(graph.number_of_nodes(), graph.number_of_edges())
    # nx.draw_networkx(graph)
    # plt.show()
    # obj = BaseGraph(graph)
    # target_k = 10
    # print(target_k, len(obj.extract_corona_nodes(obj.graph, target_k, obj.k_core)))
    total_time = 0
    for idx, node in enumerate(graph.nodes()):
        total_time += recorder[node]['time']
        if idx == int(graph.number_of_nodes() * 0.3):
            print(idx, total_time)