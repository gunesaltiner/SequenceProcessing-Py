from typing import List, Optional, Tuple

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class Im2Col1D(Function):
    """
    Rearranges a 1D input tensor (L_in, C_in) into a 2D matrix
    (L_out, kernel_size * C_in) where each row is a flattened
    sliding window over the length dimension.

    This converts a 1D convolution into a single matrix multiplication
    with a weight matrix of shape (kernel_size * C_in, C_out).
    """

    __kernel_size: int
    __stride: int
    __input_shape: Optional[Tuple[int, ...]]

    def __init__(self, kernel_size: int, stride: int = 1):
        """
        Constructor for Im2Col1D.

        :param kernel_size: Convolution kernel size.
        :param stride: Stride of the convolution.
        """
        self.__kernel_size = kernel_size
        self.__stride = stride
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

    def computeOutputLength(self, in_length: int) -> int:
        """
        Computes the output length given an input length.

        :param in_length: Input sequence length.
        :return: Output sequence length L_out = (L_in - K) // S + 1.
        """
        return (in_length - self.__kernel_size) // self.__stride + 1

    def calculate(self, matrix: Tensor) -> Tensor:
        """
        Builds the im2col matrix from the input tensor.

        :param matrix: Input tensor of shape (L_in, C_in).
        :return: Tensor of shape (L_out, kernel_size * C_in) where row i
                 is the flattened input rows [i*stride, i*stride+K).
        """
        shape = matrix.getShape()
        if len(shape) != 2:
            raise ValueError("Im2Col1D expects a 2D tensor of shape (L, C).")

        in_length = shape[0]
        in_channels = shape[1]
        out_length = self.computeOutputLength(in_length)

        if out_length <= 0:
            raise ValueError("Invalid Im2Col1D output length: kernel/stride too large.")

        self.__input_shape = shape

        out_width = self.__kernel_size * in_channels
        values = []

        for i in range(out_length):
            window_start = i * self.__stride
            for k in range(self.__kernel_size):
                row = window_start + k
                for c in range(in_channels):
                    values.append(matrix.getValue((row, c)))

        return Tensor(values, (out_length, out_width))

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Folds the backward gradient from shape (L_out, K*C_in) back into
        the original input shape (L_in, C_in). Overlapping windows have
        their gradients summed.

        :param value: Forward output tensor of shape (L_out, K*C_in).
        :param backward: Backward gradient tensor of shape (L_out, K*C_in).
        :return: Gradient tensor of shape (L_in, C_in).
        """
        if self.__input_shape is None:
            raise ValueError("Im2Col1D.calculate must be called before derivative.")

        in_length = self.__input_shape[0]
        in_channels = self.__input_shape[1]
        out_length = value.getShape()[0]

        grad_data = [0.0] * (in_length * in_channels)
        backward_data = backward.getData()

        idx = 0
        for i in range(out_length):
            window_start = i * self.__stride
            for k in range(self.__kernel_size):
                row = window_start + k
                for c in range(in_channels):
                    grad_data[row * in_channels + c] += backward_data[idx]
                    idx += 1

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

    def __repr__(self) -> str:
        """
        Returns string representation.

        :return: String representation.
        """
        return f"Im2Col1D(kernel_size={self.__kernel_size}, stride={self.__stride})"