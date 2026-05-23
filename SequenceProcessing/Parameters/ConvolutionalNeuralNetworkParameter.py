from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Initialization.Initialization import Initialization
from ComputationalGraph.NeuralNetworkParameter import NeuralNetworkParameter
from ComputationalGraph.Optimizer.Optimizer import Optimizer


class ConvolutionalNeuralNetworkParameter(NeuralNetworkParameter):
    """
    Parameter class for 1D convolutional neural networks adapted to NLP tasks.

    The architecture follows the AlexNet pattern adapted to sequences:
    a stack of 1D convolution layers (each optionally followed by a
    max-pooling layer), then a flatten step, then a stack of fully
    connected layers, and finally a softmax classifier over class_label_size
    classes.

    All conv-related lists must have the same length; all FC-related lists
    must have the same length. A pool_kernel_size of 0 means no pooling is
    applied after that convolutional layer.
    """

    __conv_filters: List[int]
    __conv_kernel_sizes: List[int]
    __conv_strides: List[int]
    __conv_activation_functions: List[Function]

    __pool_kernel_sizes: List[int]
    __pool_strides: List[int]

    __fc_hidden_layers: List[int]
    __fc_activation_functions: List[Function]

    __class_label_size: int

    def __init__(self,
                 seed: int,
                 epoch: int,
                 optimizer: Optimizer,
                 initialization: Initialization,
                 loss: Function,
                 conv_filters: List[int],
                 conv_kernel_sizes: List[int],
                 conv_strides: List[int],
                 conv_activation_functions: List[Function],
                 pool_kernel_sizes: List[int],
                 pool_strides: List[int],
                 fc_hidden_layers: List[int],
                 fc_activation_functions: List[Function],
                 class_label_size: int,
                 dropout: float = 0.0):
        """
        Constructor for ConvolutionalNeuralNetworkParameter.

        :param seed: Random seed.
        :param epoch: Number of epochs.
        :param optimizer: Optimization algorithm.
        :param initialization: Weight initialization method.
        :param loss: Loss function.
        :param conv_filters: Output channel count for each convolutional layer.
        :param conv_kernel_sizes: Kernel size for each convolutional layer.
        :param conv_strides: Stride for each convolutional layer.
        :param conv_activation_functions: Activation function for each conv layer.
        :param pool_kernel_sizes: Pool kernel for each conv layer; 0 disables pooling.
        :param pool_strides: Pool stride for each conv layer; ignored where pool kernel is 0.
        :param fc_hidden_layers: Hidden sizes of the fully connected layers (excluding the final class layer).
        :param fc_activation_functions: Activation function for each FC hidden layer.
        :param class_label_size: Number of output classes.
        :param dropout: Dropout ratio applied between FC hidden layers.
        """
        super().__init__(seed, epoch, optimizer, initialization, loss, dropout, 1)

        conv_size = len(conv_filters)
        if (len(conv_kernel_sizes) != conv_size
                or len(conv_strides) != conv_size
                or len(conv_activation_functions) != conv_size
                or len(pool_kernel_sizes) != conv_size
                or len(pool_strides) != conv_size):
            raise ValueError(
                "All convolution-related lists must have the same length."
            )

        if len(fc_hidden_layers) != len(fc_activation_functions):
            raise ValueError(
                "FC hidden layers and FC activation functions must have the same length."
            )

        if class_label_size <= 0:
            raise ValueError("class_label_size must be a positive integer.")

        for i in range(conv_size):
            if conv_filters[i] <= 0:
                raise ValueError(
                    f"conv_filters[{i}] must be positive, got {conv_filters[i]}."
                )
            if conv_kernel_sizes[i] <= 0:
                raise ValueError(
                    f"conv_kernel_sizes[{i}] must be positive, got {conv_kernel_sizes[i]}."
                )
            if conv_strides[i] <= 0:
                raise ValueError(
                    f"conv_strides[{i}] must be positive, got {conv_strides[i]}."
                )
            if pool_kernel_sizes[i] < 0:
                raise ValueError(
                    f"pool_kernel_sizes[{i}] must be non-negative, got {pool_kernel_sizes[i]}."
                )
            if pool_kernel_sizes[i] > 0 and pool_strides[i] <= 0:
                raise ValueError(
                    f"pool_strides[{i}] must be positive when pooling is enabled, got {pool_strides[i]}."
                )

        for i in range(len(fc_hidden_layers)):
            if fc_hidden_layers[i] <= 0:
                raise ValueError(
                    f"fc_hidden_layers[{i}] must be positive, got {fc_hidden_layers[i]}."
                )

        self.__conv_filters = conv_filters
        self.__conv_kernel_sizes = conv_kernel_sizes
        self.__conv_strides = conv_strides
        self.__conv_activation_functions = conv_activation_functions

        self.__pool_kernel_sizes = pool_kernel_sizes
        self.__pool_strides = pool_strides

        self.__fc_hidden_layers = fc_hidden_layers
        self.__fc_activation_functions = fc_activation_functions

        self.__class_label_size = class_label_size

    def convSize(self) -> int:
        """
        Returns the number of convolutional layers.

        :return: Number of conv layers.
        """
        return len(self.__conv_filters)

    def fcSize(self) -> int:
        """
        Returns the number of fully connected hidden layers.

        :return: Number of FC hidden layers.
        """
        return len(self.__fc_hidden_layers)

    def getConvFilter(self, index: int) -> int:
        """
        Returns the number of output filters for the conv layer at index.

        :param index: Conv layer index.
        :return: Number of output filters.
        """
        return self.__conv_filters[index]

    def getConvKernelSize(self, index: int) -> int:
        """
        Returns the kernel size of the conv layer at index.

        :param index: Conv layer index.
        :return: Kernel size.
        """
        return self.__conv_kernel_sizes[index]

    def getConvStride(self, index: int) -> int:
        """
        Returns the stride of the conv layer at index.

        :param index: Conv layer index.
        :return: Stride.
        """
        return self.__conv_strides[index]

    def getConvActivationFunction(self, index: int) -> Function:
        """
        Returns the activation function for the conv layer at index.

        :param index: Conv layer index.
        :return: Activation function.
        """
        return self.__conv_activation_functions[index]

    def getPoolKernelSize(self, index: int) -> int:
        """
        Returns the pool kernel size for the conv layer at index.
        A value of 0 indicates no pooling is applied after that layer.

        :param index: Conv layer index.
        :return: Pool kernel size, or 0 for no pooling.
        """
        return self.__pool_kernel_sizes[index]

    def getPoolStride(self, index: int) -> int:
        """
        Returns the pool stride for the conv layer at index.

        :param index: Conv layer index.
        :return: Pool stride.
        """
        return self.__pool_strides[index]

    def hasPool(self, index: int) -> bool:
        """
        Returns whether a pooling layer follows the conv layer at index.

        :param index: Conv layer index.
        :return: True if pool kernel is positive, False otherwise.
        """
        return self.__pool_kernel_sizes[index] > 0

    def getFcHiddenLayer(self, index: int) -> int:
        """
        Returns the size of the FC hidden layer at index.

        :param index: FC hidden layer index.
        :return: Hidden layer size.
        """
        return self.__fc_hidden_layers[index]

    def getFcActivationFunction(self, index: int) -> Function:
        """
        Returns the activation function for the FC hidden layer at index.

        :param index: FC hidden layer index.
        :return: Activation function.
        """
        return self.__fc_activation_functions[index]

    def getConvFilters(self) -> List[int]:
        """
        Getter for all convolutional filter counts.

        :return: List of filter counts.
        """
        return self.__conv_filters

    def getConvKernelSizes(self) -> List[int]:
        """
        Getter for all convolutional kernel sizes.

        :return: List of kernel sizes.
        """
        return self.__conv_kernel_sizes

    def getConvStrides(self) -> List[int]:
        """
        Getter for all convolutional strides.

        :return: List of strides.
        """
        return self.__conv_strides

    def getConvActivationFunctions(self) -> List[Function]:
        """
        Getter for all convolutional activation functions.

        :return: List of activation functions.
        """
        return self.__conv_activation_functions

    def getPoolKernelSizes(self) -> List[int]:
        """
        Getter for all pool kernel sizes.

        :return: List of pool kernel sizes.
        """
        return self.__pool_kernel_sizes

    def getPoolStrides(self) -> List[int]:
        """
        Getter for all pool strides.

        :return: List of pool strides.
        """
        return self.__pool_strides

    def getFcHiddenLayers(self) -> List[int]:
        """
        Getter for all FC hidden layer sizes.

        :return: List of FC hidden layer sizes.
        """
        return self.__fc_hidden_layers

    def getFcActivationFunctions(self) -> List[Function]:
        """
        Getter for all FC activation functions.

        :return: List of FC activation functions.
        """
        return self.__fc_activation_functions

    def getClassLabelSize(self) -> int:
        """
        Getter for the number of output classes.

        :return: Class label size.
        """
        return self.__class_label_size