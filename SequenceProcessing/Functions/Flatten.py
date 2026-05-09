from typing import List, Optional, Tuple

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class Flatten(Function):
    """
    Flattens a 2D tensor of shape (L, C) into a 2D row vector of shape (1, L*C).
    Used to bridge convolutional layers to fully connected layers.
    """

    __input_shape: Optional[Tuple[int, ...]]

    def __init__(self):
        """
        Constructor for Flatten. Initializes the cached input shape.
        """
        self.__input_shape = None

    def getInputShape(self) -> Optional[Tuple[int, ...]]:
        """
        Getter for the cached input shape.

        :return: Cached input shape used to undo the flatten in derivative.
        """
        return self.__input_shape

    def calculate(self, matrix: Tensor) -> Tensor:
        """
        Flattens the input tensor of shape (L, C) into shape (1, L*C).
        Caches the original shape for the backward pass.

        :param matrix: Input tensor of shape (L, C).
        :return: Flattened tensor of shape (1, L*C).
        """
        shape = matrix.getShape()
        self.__input_shape = shape

        total = 1
        for dim in shape:
            total *= dim

        return Tensor(list(matrix.getData()), (1, total))

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Reshapes the backward gradient from (1, L*C) back to (L, C).

        :param value: Forward output tensor of shape (1, L*C).
        :param backward: Backward gradient tensor of shape (1, L*C).
        :return: Gradient tensor reshaped to the original input shape (L, C).
        """
        if self.__input_shape is None:
            raise ValueError("Flatten.calculate must be called before derivative.")

        return Tensor(list(backward.getData()), self.__input_shape)

    def addEdge(self,
                input_nodes: List[ComputationalNode],
                is_biased: bool) -> ComputationalNode:
        """
        Adds this function as an edge to the computational graph.

        :param input_nodes: Input computational nodes.
        :param is_biased: Indicates whether the edge is biased.
        :return: Newly created computational node.
        """
        new_node = FunctionNode(is_biased, self)
        input_nodes[0].add(new_node)
        return new_node

    def __repr__(self) -> str:
        """
        Returns string representation.

        :return: String representation.
        """
        return "Flatten()"