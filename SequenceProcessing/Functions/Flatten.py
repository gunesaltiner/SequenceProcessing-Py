from typing import List, Optional, Tuple

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class Flatten(Function):
    """
    Flattens a 2D tensor of shape (L, C) into a 2D row vector of shape (1, L*C).
    Used to bridge convolutional layers to fully connected layers.

    Example:
        Input shape (3, 4) with 3 tokens and 4 channels:
            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12]]

        Output shape (1, 12):
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]

    The original input shape is cached during forward pass so that the
    derivative can reshape the gradient back to the correct dimensions.
    """

    # Cached input shape from the forward pass, needed by derivative to
    # restore the gradient to the original (L, C) shape.
    __input_shape: Optional[Tuple[int, ...]]

    def __init__(self):
        """
        Constructor for Flatten. Initializes the cached input shape to None.
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
        # Save the original shape so derivative() can reshape gradient back
        shape = matrix.getShape()
        self.__input_shape = shape

        # Calculate total number of elements by multiplying all dimensions.
        # For a (L, C) tensor, total = L * C.
        total = 1
        for dim in shape:
            total *= dim

        # Create a new tensor with the same data but reshaped to (1, total).
        # The (1, ...) row-vector format matches the FC layer input convention.
        return Tensor(list(matrix.getData()), (1, total))

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Reshapes the backward gradient from (1, L*C) back to (L, C).

        The gradient arrives as a flat row (1, L*C) from the FC layer above.
        We must reshape it back to (L, C) so the convolutional layer below
        receives gradients in the correct spatial layout.

        :param value: Forward output tensor of shape (1, L*C).
        :param backward: Backward gradient tensor of shape (1, L*C).
        :return: Gradient tensor reshaped to the original input shape (L, C).
        """
        # Safety check: calculate() must run before derivative()
        if self.__input_shape is None:
            raise ValueError("Flatten.calculate must be called before derivative.")

        # Simply reshape the flat gradient back to the original (L, C) shape.
        # No arithmetic is needed because flatten is just a rearrangement.
        return Tensor(list(backward.getData()), self.__input_shape)

    def addEdge(self,
                input_nodes: List[ComputationalNode],
                is_biased: bool) -> ComputationalNode:
        """
        Adds this function as an edge to the computational graph.
        Creates a FunctionNode wrapping this Flatten and connects it
        as a child of the input node.

        :param input_nodes: Input computational nodes.
        :param is_biased: Indicates whether the edge is biased.
        :return: Newly created computational node.
        """
        # Wrap this Function in a FunctionNode and link it to the parent
        new_node = FunctionNode(is_biased, self)
        input_nodes[0].add(new_node)
        return new_node

    def toString(self) -> str:
        """
        Returns string representation.

        :return: String representation.
        """
        return "Flatten()"