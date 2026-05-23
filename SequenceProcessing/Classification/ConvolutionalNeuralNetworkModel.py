from typing import List
import random

from ComputationalGraph.ComputationalGraph import ComputationalGraph
from ComputationalGraph.Function.Dropout import Dropout
from ComputationalGraph.Function.Softmax import Softmax
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.MultiplicationNode import MultiplicationNode

from Math.Tensor import Tensor

from SequenceProcessing.Functions.Im2Col1D import Im2Col1D
from SequenceProcessing.Functions.MaxPool1D import MaxPool1D
from SequenceProcessing.Functions.Flatten import Flatten
from SequenceProcessing.Parameters.ConvolutionalNeuralNetworkParameter import (
    ConvolutionalNeuralNetworkParameter,
)


class ConvolutionalNeuralNetworkModel(ComputationalGraph):
    """
    1D convolutional neural network for sentence-level classification,
    structured as an AlexNet adaptation for NLP.

    The architecture is built from the ConvolutionalNeuralNetworkParameter:
    a stack of conv layers (Im2Col1D + learnable weight matrix + activation,
    each optionally followed by MaxPool1D), then a Flatten, then a stack of
    fully connected hidden layers, and finally a softmax classifier over
    class_label_size classes.

    Each training instance is a flat tensor of length T * E + 1, where
    T = max_sequence_length, E = word_embedding_length, and the final value
    is the integer class label of the whole sentence. Sentences shorter
    than T are zero-padded.

    Both convolutional and fully connected layers carry an additive bias
    via the framework's is_biased mechanism.
    """

    __word_embedding_length: int
    __max_sequence_length: int

    def __init__(self,
                 parameter: ConvolutionalNeuralNetworkParameter,
                 word_embedding_length: int,
                 max_sequence_length: int):
        """
        Constructor for ConvolutionalNeuralNetworkModel.

        :param parameter: Convolutional neural network parameters.
        :param word_embedding_length: Word embedding (input channel) size.
        :param max_sequence_length: Maximum number of tokens per sentence.
        """
        super().__init__(parameter)
        self.__word_embedding_length = word_embedding_length
        self.__max_sequence_length = max_sequence_length

    def getWordEmbeddingLength(self) -> int:
        """
        Getter for word embedding length.

        :return: Word embedding length.
        """
        return self.__word_embedding_length

    def getMaxSequenceLength(self) -> int:
        """
        Getter for max sequence length.

        :return: Maximum number of tokens per sentence.
        """
        return self.__max_sequence_length

    def computeConvOutputLength(self, in_length: int, kernel_size: int, stride: int) -> int:
        """
        Computes the output length of a 1D convolution or pool layer.

        :param in_length: Input sequence length.
        :param kernel_size: Kernel size.
        :param stride: Stride.
        :return: Output sequence length.
        """
        return (in_length - kernel_size) // stride + 1

    def createInputTensor(self, instance: Tensor) -> Tensor:
        """
        Reshapes the flat instance tensor into a 2D matrix of shape
        (max_sequence_length, word_embedding_length), zero-padding any
        missing positions.

        :param instance: Flat tensor of length T * E + 1 where the last
                         value is the class label.
        :return: 2D tensor of shape (max_sequence_length, word_embedding_length).
        """
        embedding_length = self.__word_embedding_length
        max_length = self.__max_sequence_length
        instance_length = instance.getShape()[0] - 1
        provided_tokens = instance_length // embedding_length

        values = []

        for i in range(max_length):
            if i < provided_tokens:
                base = i * embedding_length
                for j in range(embedding_length):
                    values.append(instance.getValue((base + j,)))
            else:
                for _ in range(embedding_length):
                    values.append(0.0)

        return Tensor(values, (max_length, embedding_length))

    def createClassLabelTensor(self, class_label: int) -> Tensor:
        """
        Creates a one-hot row tensor for the given class label.

        :param class_label: Integer class label.
        :return: One-hot tensor of shape (1, class_label_size).
        """
        class_label_size = self.parameters.getClassLabelSize()
        values = [1.0 if i == class_label else 0.0 for i in range(class_label_size)]
        return Tensor(values, (1, class_label_size))

    def extractClassLabel(self, instance: Tensor) -> int:
        """
        Extracts the integer class label from the last position of the
        flat instance tensor.

        :param instance: Flat instance tensor.
        :return: Integer class label.
        """
        last_index = instance.getShape()[0] - 1
        return int(instance.getValue((last_index,)))

    def buildGraph(self, random_generator: random.Random) -> ComputationalNode:
        """
        Builds the full computational graph (conv block + FC block + softmax)
        and returns the class-label input node so the training loop can set
        gold labels on it.

        :param random_generator: Random generator for weight initialization.
        :return: Class-label input node.
        """
        params: ConvolutionalNeuralNetworkParameter = self.parameters


        # Input node: a 2D tensor (max_sequence_length, word_embedding_length)
        input_node = MultiplicationNode(False, False)
        self.input_nodes.append(input_node)

        current = input_node
        current_length = self.__max_sequence_length
        current_channels = self.__word_embedding_length

        # Convolutional block
        # In the AlexNet-NLP diagram these correspond to Conv1 through Conv5:
        #   conv layer 0 = Conv1 (kernel=5, 96 filters, pool after)
        #   conv layer 1 = Conv2 (kernel=5, 256 filters, pool after)
        #   conv layer 2 = Conv3 (kernel=3, 384 filters, no pool)
        #   conv layer 3 = Conv4 (kernel=3, 384 filters, no pool)
        #   conv layer 4 = Conv5 (kernel=3, 256 filters, pool after)
        #
        # Each conv layer is Im2Col1D(is_biased=True) -> MultiplicationNode(W)
        # -> activation, optionally followed by MaxPool1D.
        # The is_biased flag on Im2Col1D appends a bias column, turning the
        # output shape from (L_out, K*C_in) into (L_out, K*C_in + 1).
        # The conv weight matrix absorbs the bias as its last row.
        for i in range(params.convSize()):
            kernel_size = params.getConvKernelSize(i)
            stride = params.getConvStride(i)
            out_channels = params.getConvFilter(i)

            im2col_out = self.addEdge(current, Im2Col1D(kernel_size, stride), True)
            current_length = self.computeConvOutputLength(current_length, kernel_size, stride)

            in_features_with_bias = kernel_size * current_channels + 1
            conv_weight = MultiplicationNode(
                Tensor(
                    params.initializeWeights(
                        in_features_with_bias, out_channels, random_generator
                    ),
                    (in_features_with_bias, out_channels),
                )
            )
            conv_out = self.addEdge(im2col_out, conv_weight)
            current_channels = out_channels

            current = self.addEdge(conv_out, params.getConvActivationFunction(i))

            if params.hasPool(i):
                pool_kernel = params.getPoolKernelSize(i)
                pool_stride = params.getPoolStride(i)
                current = self.addEdge(current, MaxPool1D(pool_kernel, pool_stride))
                current_length = self.computeConvOutputLength(
                    current_length, pool_kernel, pool_stride
                )

        # Flatten conv output (L, C) -> (1, L*C). Set is_biased=True so framework appends a bias column for the first FC layer
        current = self.addEdge(current, Flatten(), True)
        flatten_features = current_length * current_channels

        # Fully connected hidden layers with Dropout.
        #
        # In the AlexNet-NLP diagram these correspond to:
        #   fc_hidden_layers[0] = FC6 (4096 units)
        #   fc_hidden_layers[1] = FC7 (4096 units)
        #
        # Each FC hidden layer is: MultiplicationNode(W) -> activation -> Dropout
        # Dropout is applied between FC hidden layers to prevent overfitting,
        # matching standard AlexNet practice. The dropout ratio comes from the
        # parameter object (default 0.0 means no dropout).
        prev_features_with_bias = flatten_features + 1
        dropout_ratio = params.getDropout()

        for i in range(params.fcSize()):
            hidden_size = params.getFcHiddenLayer(i)

            fc_weight = MultiplicationNode(
                Tensor(
                    params.initializeWeights(
                        prev_features_with_bias, hidden_size, random_generator
                    ),
                    (prev_features_with_bias, hidden_size),
                )
            )
            fc_out = self.addEdge(current, fc_weight)

            # If Dropout will follow, activation is NOT biased (Dropout adds bias instead).
            # If no Dropout, activation itself adds the bias for the next layer.
            if dropout_ratio > 0.0:
                current = self.addEdge(fc_out, params.getFcActivationFunction(i), False)
                current = self.addEdge(
                    current,
                    Dropout(dropout_ratio, random.Random(params.getSeed() + i)),
                    True
                )
            else:
                current = self.addEdge(fc_out, params.getFcActivationFunction(i), True)

            prev_features_with_bias = hidden_size + 1

        # Final classification head (FC8 in the AlexNet-NLP diagram):
        # dense layer mapping to class_label_size outputs, followed by Softmax.
        class_label_size = params.getClassLabelSize()
        output_weight = MultiplicationNode(
            Tensor(
                params.initializeWeights(
                    prev_features_with_bias, class_label_size, random_generator
                ),
                (prev_features_with_bias, class_label_size),
            )
        )
        logits = self.addEdge(current, output_weight)
        self.output_node = self.addEdge(logits, Softmax())

        # Class-label input node + loss edge
        class_label_node = ComputationalNode()
        self.input_nodes.append(class_label_node)

        loss_inputs = [self.output_node, class_label_node]
        self.addFunctionEdge(loss_inputs, params.getLossFunction(), False)

        return class_label_node

    def trainInternal(self,
                      train_set: List[Tensor],
                      class_label_node: ComputationalNode,
                      random_generator: random.Random) -> None:
        """
        Internal training loop: shuffle, set inputs and labels, forward, backward.

        :param train_set: Training instances (flat tensors).
        :param class_label_node: Class-label input node returned by buildGraph.
        :param random_generator: Random generator used for shuffling.
        """
        params: ConvolutionalNeuralNetworkParameter = self.parameters
        input_node = self.input_nodes[0]

        for _ in range(params.getEpoch()):
            for _ in range(len(train_set)):
                i1 = random_generator.randint(0, len(train_set) - 1)
                i2 = random_generator.randint(0, len(train_set) - 1)
                train_set[i1], train_set[i2] = train_set[i2], train_set[i1]

            for instance in train_set:
                input_node.setValue(self.createInputTensor(instance))

                class_label = self.extractClassLabel(instance)
                class_label_node.setValue(self.createClassLabelTensor(class_label))

                self.forwardCalculation()
                self.backpropagation()

            params.getOptimizer().setLearningRate()

    def train(self, train_set: List[Tensor]) -> None:
        """
        Builds the computational graph and runs the full training loop.

        :param train_set: Training set of flat instance tensors.
        """
        random_generator = random.Random(self.parameters.getSeed())
        class_label_node = self.buildGraph(random_generator)
        self.trainInternal(train_set, class_label_node, random_generator)

    def getOutputValue(self, output_node: ComputationalNode) -> List[float]:
        """
        Extracts the predicted class index from the softmax output.

        :param output_node: Output computational node holding softmax probabilities.
        :return: Single-element list with the argmax class index as a float.
        """
        output_value = output_node.getValue()
        if output_value is None:
            return []

        shape = output_value.getShape()
        cols = shape[1]

        max_val = float("-inf")
        max_index = -1

        for j in range(cols):
            val = output_value.getValue((0, j))
            if val > max_val:
                max_val = val
                max_index = j

        return [float(max_index)]

    def test(self, test_set: List[Tensor]) -> float:
        """
        Evaluates the model on the test set and returns accuracy.

        :param test_set: Test set of flat instance tensors.
        :return: Accuracy as a float in [0, 1].
        """
        input_node = self.input_nodes[0]
        count = 0
        total = 0

        for instance in test_set:
            input_node.setValue(self.createInputTensor(instance))
            predicted = int(self.predict()[0])

            if predicted == self.extractClassLabel(instance):
                count += 1
            total += 1

        if total == 0:
            return 0.0
        return count / total