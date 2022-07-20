import numpy as np

from typing import Tuple, List
from functional.utils.box import clip_bboxes
from .text_proposal_graph_builder import TextProposalGraphBuilder


def fit_y(X: np.ndarray, Y: np.ndarray, x1: np.ndarray, x2: np.ndarray):
    """
    Interpolate the vertical coordinates based on data and 2 given horizontal coordinates.

    Args:
        X (numpy array): A numpy array contains the horizontal coordinates.
        Y (numpy array): A numpy array contains the vertical coordinates.
        x1 (numpy array): The horizontal coordinate of point 1.
        x2 (numpy array): The horizontal coordinate of point 2.

    Returns:
        An interpolation of the vertical coordinates.

    """
    # if X only include one point, the function will get line2Match y=Y[0]
    if np.sum(X == X[0]) == len(X):
        return Y[0], Y[0]
    p = np.poly1d(np.polyfit(X, Y, 1))
    return p(x1), p(x2)


class TextProposalConnector(object):
    def __init__(self, configs: dict):
        """
        Connect text proposals into text bouding boxes.
        
        Args:
            configs (dict): The configuration file.
            
        """

        self.graph_builder: object = TextProposalGraphBuilder(configs)

    def group_text_proposals(self,
                             text_proposals: np.ndarray,
                             scores: np.ndarray,
                             im_size: Tuple[int, int]) -> List[List[int]]:
        """
        Group text proposals into groups. Each group contains the text proposals belong into the same line of text.
        
        Args:
            text_proposals (numpy array): A Numpy array that contains the coordinates of each text proposal.
            scores (numpy array): A Numpy array that contains the predicted confidence of each text proposal.
            im_size (int, tuple): The image's size.

        Returns:
            A group of the text proposals.
            
        """

        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)

        return graph.sub_graphs_connected()

    def get_text_lines(self,
                       text_proposals: np.ndarray,
                       scores: np.ndarray,
                       im_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine all text proposals into bounding boxes.
        
        Args:
            text_proposals (numpy array): A Numpy array that contains the coodinates of each text proposal.
            scores (numpy array): A Numpy array that contains the predicted confidence of each text proposal.
            im_size (int, tuple): The image's size.

        Returns:
            The bounding boxes and scores for each line.
            
        """
        # Group text proposals
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)

        # Initialize the list of text bounding boxes and scores.
        text_lines = np.zeros((len(tp_groups), 4), dtype=np.float32)
        average_scores = []

        # Now, connect the text proposals in each group
        for index, tp_indices in enumerate(tp_groups):
            # Get the coordinates, offset, and scores of each proposal in group.
            text_line_boxes = text_proposals[list(tp_indices)]

            # Get the predicted top left and bottom right x-coordinates of the text lines.
            xmin = np.min(text_line_boxes[:, 0])
            xmax = np.max(text_line_boxes[:, 2])

            # Find vertical coordinates of text lines
            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) / 2.
            lt_y, rt_y = fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], xmin + offset, xmax - offset)
            lb_y, rb_y = fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], xmin + offset, xmax - offset)

            # the score of a text line is the average score of the scores
            # of all text proposals contained in the text line.
            average_scores.append(scores[list(tp_indices)].sum() / float(len(tp_indices)))

            # Appending the bounding boxes coordinates and scores.
            text_lines[index, 0] = xmin
            text_lines[index, 1] = min(lt_y, rt_y)
            text_lines[index, 2] = xmax
            text_lines[index, 3] = max(lb_y, rb_y)

        # Keep bounding boxes inside the image size.
        text_lines = clip_bboxes(text_lines, im_size)

        average_scores = np.array(average_scores)

        return text_lines, average_scores
