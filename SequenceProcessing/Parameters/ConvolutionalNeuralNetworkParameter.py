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

    For the AlexNet-NLP diagram, the mapping is:
        conv_filters       = [96, 256, 384, 384, 256]  -> Conv1..Conv5
        conv_kernel_sizes   = [5, 5, 3, 3, 3]
        pool_kernel_sizes   = [3, 3, 0, 0, 3]          -> 0 means no pool
        fc_hidden_layers    = [4096, 4096]              -> FC6 and FC7
        class_label_size    = N                          -> FC8 (output layer)

    All conv-related lists must have the same length; all FC-related lists
    must have the same length. A pool_kernel_size of 0 means no pooling is
    applied after that convolutional layer.
    """

    # Output channel count for each conv layer (e.g. [96, 256, 384, 384, 256])
    __conv_filters: List[int]
    # Kernel size for each conv layer (e.g. [5, 5, 3, 3, 3])
    __conv_kernel_sizes: List[int]
    # Stride for each conv layer (e.g. [1, 1, 1, 1, 1])
    __conv_strides: List[int]
    # Activation function for each conv layer (e.g. ReLU for all)
    __conv_activation_functions: List[Function]

    # Pool kernel size for each conv layer; 0 means no pooling after that layer
    __pool_kernel_sizes: List[int]
    # Pool stride for each conv layer; ignored where pool kernel is 0
    __pool_strides: List[int]

    # Hidden sizes for FC layers, excluding the final classification layer.
    # For AlexNet-NLP: [4096, 4096] -> FC6 and FC7
    __fc_hidden_layers: List[int]
    # Activation function for each FC hidden layer
    __fc_activation_functions: List[Function]

    # Number of output classes for the final classification layer (FC8)
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
        # Pass common training parameters to the parent class.
        # batch_size is set to 1 (online learning, same as RNN model).
        super().__init__(seed, epoch, optimizer, initialization, loss, dropout, 1)

        # ---- Validate that all conv-related lists have the same length ----
        conv_size = len(conv_filters)
        if (len(conv_kernel_sizes) != conv_size
                or len(conv_strides) != conv_size
                or len(conv_activation_functions) != conv_size
                or len(pool_kernel_sizes) != conv_size
                or len(pool_strides) != conv_size):
            raise ValueError(
                "All convolution-related lists must have the same length."
            )

        # ---- Validate that FC lists match in length ----
        if len(fc_hidden_layers) != len(fc_activation_functions):
            raise ValueError(
                "FC hidden layers and FC activation functions must have the same length."
            )

        # ---- Validate class label size is positive ----
        if class_label_size <= 0:
            raise ValueError("class_label_size must be a positive integer.")

        # ---- Validate each conv layer's hyperparameters individually ----
        # Catches errors at parameter creation time rather than during training
        for i in range(conv_size):
            # Filter count must be positive (number of output channels)
            if conv_filters[i] <= 0:
                raise ValueError(
                    f"conv_filters[{i}] must be positive, got {conv_filters[i]}."
                )
            # Kernel size must be positive (width of the sliding window)
            if conv_kernel_sizes[i] <= 0:
                raise ValueError(
                    f"conv_kernel_sizes[{i}] must be positive, got {conv_kernel_sizes[i]}."
                )
            # Stride must be positive (step size between windows)
            if conv_strides[i] <= 0:
                raise ValueError(
                    f"conv_strides[{i}] must be positive, got {conv_strides[i]}."
                )
            # Pool kernel must be non-negative (0 means no pooling)
            if pool_kernel_sizes[i] < 0:
                raise ValueError(
                    f"pool_kernel_sizes[{i}] must be non-negative, got {pool_kernel_sizes[i]}."
                )
            # If pooling is enabled, pool stride must be positive
            if pool_kernel_sizes[i] > 0 and pool_strides[i] <= 0:
                raise ValueError(
                    f"pool_strides[{i}] must be positive when pooling is enabled, got {pool_strides[i]}."
                )

        # ---- Validate each FC hidden layer size is positive ----
        for i in range(len(fc_hidden_layers)):
            if fc_hidden_layers[i] <= 0:
                raise ValueError(
                    f"fc_hidden_layers[{i}] must be positive, got {fc_hidden_layers[i]}."
                )

        # Store all validated parameters
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
        Note: this does NOT count the final classification layer (FC8),
        which is added by the model automatically.

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
        Pooling is enabled when pool_kernel_size > 0.

        :param index: Conv layer index.
        :return: True if pool kernel is positive, False otherwise.
        """
        return self.__pool_kernel_sizes[index] > 0

    def getFcHiddenLayer(self, index: int) -> int:
        """
        Returns the size of the FC hidden layer at index.
        For AlexNet-NLP: index 0 = FC6 (4096), index 1 = FC7 (4096).

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
        This corresponds to FC8 in the AlexNet-NLP diagram.

        :return: Class label size.
        """
        return self.__class_label_size