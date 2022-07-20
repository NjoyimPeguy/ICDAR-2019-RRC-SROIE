import numpy as np

from typing import List


class Graph:
    def __init__(self, graph: object):
        """
        Object represents the graph containing the connected text proposals.
        
        Args:
            graph (object): The graph object.
            
        """
        self.graph: object = graph

    def sub_graphs_connected(self) -> List[List[int]]:
        """
        Refine the original graph having num_proposals x num_proposals vertices
        into a list of group of connected text proposals
        
        Returns:
            A sub-graph, i.e., a list of indexes of connected text proposals.
            
        """
        sub_graphs = []
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v = index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs
