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

    Example with input (4, 2), kernel_size=3, stride=1:

        Input (4 tokens, 2 channels):
            token 0: [1, 2]
            token 1: [3, 4]
            token 2: [5, 6]
            token 3: [7, 8]

        Window 0 covers tokens [0, 1, 2] -> flattened: [1, 2, 3, 4, 5, 6]
        Window 1 covers tokens [1, 2, 3] -> flattened: [3, 4, 5, 6, 7, 8]

        Output (2, 6):
            [[1, 2, 3, 4, 5, 6],
             [3, 4, 5, 6, 7, 8]]

    After this rearrangement, multiplying by a weight matrix of shape
    (6, out_channels) produces the same result as a 1D convolution.

    Why Im2Col instead of direct convolution?
        The ComputationalGraph framework only has MultiplicationNode for
        learnable weights. Im2Col converts convolution into matrix
        multiplication so we can reuse the existing weight update mechanism.
    """

    __kernel_size: int
    __stride: int
    # Cached input shape from the forward pass, needed by derivative
    # to redistribute gradients back to the correct input positions.
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
        Formula: L_out = (L_in - kernel_size) // stride + 1

        :param in_length: Input sequence length.
        :return: Output sequence length.
        """
        return (in_length - self.__kernel_size) // self.__stride + 1

    def calculate(self, matrix: Tensor) -> Tensor:
        """
        Builds the im2col matrix from the input tensor (forward pass).

        For each output position i, extracts a window of kernel_size
        consecutive rows starting at position i * stride, flattens the
        window into a single row of length kernel_size * C_in.

        :param matrix: Input tensor of shape (L_in, C_in).
        :return: Tensor of shape (L_out, kernel_size * C_in) where row i
                 is the flattened input rows [i*stride, i*stride+K).
        """
        # Validate that input is 2D (sequence_length, channels)
        shape = matrix.getShape()
        if len(shape) != 2:
            raise ValueError("Im2Col1D expects a 2D tensor of shape (L, C).")

        in_length = shape[0]
        in_channels = shape[1]
        out_length = self.computeOutputLength(in_length)

        if out_length <= 0:
            raise ValueError("Invalid Im2Col1D output length: kernel/stride too large.")

        # Cache input shape for the backward pass (derivative needs it)
        self.__input_shape = shape

        # Each output row has kernel_size * in_channels elements
        out_width = self.__kernel_size * in_channels
        values = []

        # Outer loop: iterate over each output position (each sliding window)
        for i in range(out_length):
            # Calculate where this window starts in the input
            window_start = i * self.__stride

            # Middle loop: iterate over each row within the window
            for k in range(self.__kernel_size):
                row = window_start + k

                # Inner loop: iterate over each channel in the current row
                # and append the value to the flattened output row
                for c in range(in_channels):
                    values.append(matrix.getValue((row, c)))

        return Tensor(values, (out_length, out_width))

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Folds the backward gradient from shape (L_out, K*C_in) back into
        the original input shape (L_in, C_in).

        Key insight: during forward pass, one input position can appear in
        multiple sliding windows (when stride < kernel_size). Therefore,
        in the backward pass, that input position receives gradients from
        ALL windows it appeared in. We use += (accumulate) instead of =
        (overwrite) to sum these overlapping gradients correctly.

        Example: input length=4, kernel=3, stride=1
            Input row 0 appears in 1 window  -> gets 1x gradient
            Input row 1 appears in 2 windows -> gets 2x gradient (summed)
            Input row 2 appears in 2 windows -> gets 2x gradient (summed)
            Input row 3 appears in 1 window  -> gets 1x gradient

        :param value: Forward output tensor of shape (L_out, K*C_in).
        :param backward: Backward gradient tensor of shape (L_out, K*C_in).
        :return: Gradient tensor of shape (L_in, C_in).
        """
        # Safety check: calculate() must run before derivative()
        if self.__input_shape is None:
            raise ValueError("Im2Col1D.calculate must be called before derivative.")

        in_length = self.__input_shape[0]
        in_channels = self.__input_shape[1]
        out_length = value.getShape()[0]

        # Initialize gradient for every input position to 0.
        # We will accumulate into this array.
        grad_data = [0.0] * (in_length * in_channels)
        backward_data = backward.getData()

        # idx tracks our position in the flat backward gradient array
        idx = 0

        # Outer loop: iterate over each output position (each window)
        for i in range(out_length):
            window_start = i * self.__stride

            # Middle loop: iterate over each row within the window
            for k in range(self.__kernel_size):
                row = window_start + k

                # Inner loop: iterate over each channel
                # Use += to accumulate gradients from overlapping windows.
                # If this input position appeared in multiple windows,
                # it receives gradient from each one.
                for c in range(in_channels):
                    grad_data[row * in_channels + c] += backward_data[idx]
                    idx += 1

        return Tensor(grad_data, self.__input_shape)

    def addEdge(self,
                input_nodes: List[ComputationalNode],
                is_biased: bool) -> ComputationalNode:
        """
        Adds this function as an edge to the computational graph.
        Creates a FunctionNode wrapping this Im2Col1D and connects it
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
        return f"Im2Col1D(kernel_size={self.__kernel_size}, stride={self.__stride})"