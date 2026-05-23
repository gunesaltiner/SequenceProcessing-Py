from typing import List, Optional, Tuple

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class MaxPool1D(Function):
    """
    1D max pooling over the length (token) dimension.

    For input shape (L_in, C), produces output shape (L_out, C) where
    L_out = (L_in - kernel_size) // stride + 1. Each output position
    holds the maximum value within its window for each channel.
    """

    __kernel_size: int
    __stride: int
    __arg_max: Optional[List[int]]
    __input_shape: Optional[Tuple[int, ...]]

    def __init__(self, kernel_size: int, stride: int = 1):
        """
        Constructor for MaxPool1D.

        :param kernel_size: Size of the pooling window.
        :param stride: Stride between successive windows.
        """
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__arg_max = None
        self.__input_shape = None

    def getKernelSize(self) -> int:
        """
        Getter for kernel size.

        :return: Kernel size.
        """
        return self.__kernel_size

    def getStride(self) -> int:
        """
        Getter for stride.

        :return: Stride.
        """
        return self.__stride

    def calculate(self, matrix: Tensor) -> Tensor:
        """
        Computes the forward max pooling pass over the length dimension.
        Caches argmax indices for the backward pass.

        :param matrix: Input tensor of shape (L_in, C).
        :return: Pooled tensor of shape (L_out, C).
        """
        shape = matrix.getShape()
        if len(shape) != 2:
            raise ValueError("MaxPool1D expects a 2D tensor of shape (L, C).")

        in_length = shape[0]
        channels = shape[1]
        out_length = (in_length - self.__kernel_size) // self.__stride + 1

        if out_length <= 0:
            raise ValueError("Invalid MaxPool1D output length: kernel/stride too large.")

        self.__input_shape = shape
        self.__arg_max = [0] * (out_length * channels)

        values = []

        for i in range(out_length):
            window_start = i * self.__stride
            for c in range(channels):
                max_val = float("-inf")
                max_idx = window_start
                for k in range(self.__kernel_size):
                    pos = window_start + k
                    candidate = matrix.getValue((pos, c))
                    if candidate > max_val:
                        max_val = candidate
                        max_idx = pos
                values.append(max_val)
                self.__arg_max[i * channels + c] = max_idx

        return Tensor(values, (out_length, channels))

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Routes the backward gradient only to the input positions that
        produced the max during the forward pass.

        :param value: Forward output tensor of shape (L_out, C).
        :param backward: Backward gradient tensor of shape (L_out, C).
        :return: Gradient tensor of shape (L_in, C).
        """
        if self.__arg_max is None or self.__input_shape is None:
            raise ValueError("MaxPool1D.calculate must be called before derivative.")

        in_length = self.__input_shape[0]
        channels = self.__input_shape[1]
        out_length = value.getShape()[0]

        grad_data = [0.0] * (in_length * channels)
        backward_data = backward.getData()

        for i in range(out_length):
            for c in range(channels):
                argmax_pos = self.__arg_max[i * channels + c]
                grad_data[argmax_pos * channels + c] += backward_data[i * channels + c]

        return Tensor(grad_data, self.__input_shape)

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

    def toString(self) -> str:
        """
        Returns string representation.

        :return: String representation.
        """
        return f"MaxPool1D(kernel_size={self.__kernel_size}, stride={self.__stride})"