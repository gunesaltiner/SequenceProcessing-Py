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

    Example with input (4, 2), kernel_size=2, stride=2:

        Input (4 tokens, 2 channels):
            token 0: [1, 2]
            token 1: [3, 4]
            token 2: [2, 5]
            token 3: [6, 1]

        Window 0 (tokens 0-1): max per channel = [max(1,3), max(2,4)] = [3, 4]
        Window 1 (tokens 2-3): max per channel = [max(2,6), max(5,1)] = [6, 5]

        Output (2, 2):
            [[3, 4],
             [6, 5]]

    Why save argmax indices?
        During backpropagation, the gradient must flow ONLY to the input
        position that held the maximum value. All other positions in the
        window get zero gradient. We must remember which position was the
        max during forward pass to route the gradient correctly.
    """

    __kernel_size: int
    __stride: int
    # Stores the argmax position (input row index) for each output element.
    # Needed by derivative() to route gradients to the correct positions.
    __arg_max: Optional[List[int]]
    # Cached input shape for reconstructing the gradient tensor.
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
        For each window and each channel, finds the maximum value and
        records which input position held that maximum (argmax).

        :param matrix: Input tensor of shape (L_in, C).
        :return: Pooled tensor of shape (L_out, C).
        """
        # Validate that input is 2D (sequence_length, channels)
        shape = matrix.getShape()
        if len(shape) != 2:
            raise ValueError("MaxPool1D expects a 2D tensor of shape (L, C).")

        in_length = shape[0]
        channels = shape[1]
        out_length = (in_length - self.__kernel_size) // self.__stride + 1

        if out_length <= 0:
            raise ValueError("Invalid MaxPool1D output length: kernel/stride too large.")

        # Cache input shape and allocate argmax storage for backward pass
        self.__input_shape = shape
        self.__arg_max = [0] * (out_length * channels)

        values = []

        # Outer loop: iterate over each output position (each pooling window)
        for i in range(out_length):
            # Calculate where this pooling window starts in the input
            window_start = i * self.__stride

            # Inner loop: for each channel, find the max value in this window
            for c in range(channels):
                max_val = float("-inf")
                max_idx = window_start

                # Scan all positions within the pooling window
                for k in range(self.__kernel_size):
                    pos = window_start + k
                    candidate = matrix.getValue((pos, c))

                    # Update max if we found a larger value
                    if candidate > max_val:
                        max_val = candidate
                        max_idx = pos

                # Store the max value as output
                values.append(max_val)

                # Save which input position was the max (argmax).
                # derivative() will use this to route gradient correctly.
                self.__arg_max[i * channels + c] = max_idx

        return Tensor(values, (out_length, channels))

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Routes the backward gradient only to the input positions that
        produced the max during the forward pass. All other positions
        receive zero gradient.

        When windows overlap (stride < kernel_size), one input position
        can be the argmax in multiple windows. In that case, gradients
        from all those windows are summed using += (accumulate).

        Example: if input position 2 was the max in windows 0, 1, and 2,
        it receives the sum of all three windows' gradients.

        :param value: Forward output tensor of shape (L_out, C).
        :param backward: Backward gradient tensor of shape (L_out, C).
        :return: Gradient tensor of shape (L_in, C).
        """
        # Safety check: calculate() must run before derivative()
        if self.__arg_max is None or self.__input_shape is None:
            raise ValueError("MaxPool1D.calculate must be called before derivative.")

        in_length = self.__input_shape[0]
        channels = self.__input_shape[1]
        out_length = value.getShape()[0]

        # Initialize all input gradients to zero.
        # Only argmax positions will receive non-zero gradient.
        grad_data = [0.0] * (in_length * channels)
        backward_data = backward.getData()

        # For each output position and channel, route the gradient
        # to the input position that was the argmax during forward pass
        for i in range(out_length):
            for c in range(channels):
                # Look up which input position was the max for this output
                argmax_pos = self.__arg_max[i * channels + c]

                # Send the gradient to that position.
                # Use += to accumulate if this position is argmax in
                # multiple overlapping windows.
                grad_data[argmax_pos * channels + c] += backward_data[i * channels + c]

        return Tensor(grad_data, self.__input_shape)

    def addEdge(self,
                input_nodes: List[ComputationalNode],
                is_biased: bool) -> ComputationalNode:
        """
        Adds this function as an edge to the computational graph.
        Creates a FunctionNode wrapping this MaxPool1D and connects it
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
        return f"MaxPool1D(kernel_size={self.__kernel_size}, stride={self.__stride})"