import numpy as np

from .graph import Graph
from typing import Tuple, List


class TextProposalGraphBuilder(object):

    def __init__(self, configs: dict):
        """
        Build text proposals into a graph.
        
        Args:
            configs (dict): The config path_to_file.
            
        """
        self.configs: dict = configs

    def get_successions(self, index: int) -> List[int]:
        """
        Find text proposals belonging to same group of the current text proposal.
        
        Args:
            index (int): The id of current vertex.

        Returns:
            List of integer contains the index of suitable text proposals.
            
        """
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) + 1,
                          min(int(box[0]) + self.configs.TEXTLINE.MAX_HORIZONTAL_GAP + 1, self.im_size[1])):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def get_precursors(self, index: int) -> List[int]:
        """
        Get the previous (right-side) text proposals belonging to the same group of the current text proposals.
        
        Args:
            index (int): The id of current vertex.

        Returns:
            List of integer contains the index of suitable text proposals.
            
        """
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) - 1, max(int(box[0] - self.configs.TEXTLINE.MAX_HORIZONTAL_GAP), 0) - 1, -1):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def is_succession_node(self, index: int, succession_index: int) -> bool:
        """
        Check if a provided text proposal is connected to the current text proposal.
        
        Args:
            index (int): The ID of current vertex.
            succession_index (int): The ID of the next vertex.

        Returns:
            A boolean indication whether a given text proposal is connected to current text proposal.
            
        """

        # Get all right-side text proposals belonging to same group
        precursors = self.get_precursors(succession_index)

        # If text proposal having higher or equal score than right-side text proposals, return True.
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True

        # Otherwise False.
        return False

    def meet_v_iou(self, first_index: int, second_index: int) -> bool:
        """
        Check if two text proposals belong into same group.
        Fist, we check the vertical overlap and then check the size similarity.
        
        Args:
            first_index (int): the index of the first text proposal
            second_index (int): the index of the second text proposal

        Returns:

        """

        def vertical_overlap(first_index: int, second_index: int) -> float:
            h1 = self.heights[first_index]
            h2 = self.heights[second_index]
            y0 = max(self.text_proposals[second_index][1], self.text_proposals[first_index][1])
            y1 = min(self.text_proposals[second_index][3], self.text_proposals[first_index][3])
            return max(0, y1 - y0 + 1) / min(h1, h2)

        def size_similarity(first_index: int, second_index: int) -> float:
            h1 = self.heights[first_index]
            h2 = self.heights[second_index]
            return min(h1, h2) / max(h1, h2)

        return vertical_overlap(first_index, second_index) >= self.configs.TEXTLINE.MIN_V_OVERLAPS and \
               size_similarity(first_index, second_index) >= self.configs.TEXTLINE.MIN_SIZE_SIM

    def build_graph(self, text_proposals: np.ndarray, scores: np.ndarray, im_size: Tuple[int, int]) -> Graph:
        """
        Build graph of text proposals. This graph has num_proposals x num_proposals vertices, and vertices is connected
        if corresponding text proposals is also connected (belong in to a same text boxes).
        
        Args:
            text_proposals (np.ndarray): A Numpy array containing the coordinates of each text proposal. Shape: [N, 4]
            scores (np.ndarray): A Numpy array that contains the predicted confidence of each text proposal. Shape: [N,]
            im_size (Tuple, int): The image's size.

        Returns:
            A graph
        """
        self.text_proposals: np.ndarray = text_proposals
        self.scores: np.ndarray = scores
        self.im_size: Tuple[int, int] = im_size
        self.heights: np.ndarray = text_proposals[:, 3] - text_proposals[:, 1] + 1

        boxes_table = [[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table: List[List[int]] = boxes_table

        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            succession_index = successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions(precursors)
                # if multiple successions(precursors) have equal scores.
                graph[index, succession_index] = True

        G = Graph(graph)

        return G
