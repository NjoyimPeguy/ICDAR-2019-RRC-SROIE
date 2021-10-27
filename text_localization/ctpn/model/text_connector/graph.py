import numpy as np


class Graph:
    def __init__(self, graph):
        """
        Object represents the graph containing the connected text proposals.
        
        Args:
            graph: The graph object.
            
        """
        self.graph = graph
    
    def sub_graphs_connected(self):
        """
        Refine the original graph having num_proposals x num_proposals vertices
        into a list of group of connected text proposals
        
        Returns:
            A sub-graph, i.e., a list of list of indexes of connected text proposals.
            
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
