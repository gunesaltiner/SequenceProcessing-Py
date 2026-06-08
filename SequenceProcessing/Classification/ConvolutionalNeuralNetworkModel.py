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

    # Word embedding dimension (e.g. 300 for Word2Vec, matches C_in of Conv1)
    __word_embedding_length: int
    # Fixed input sequence length (e.g. 512 tokens, shorter sentences are padded)
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
        Formula: L_out = (L_in - kernel_size) // stride + 1

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

        The flat instance format is:
            [emb_0_0, emb_0_1, ..., emb_0_E, emb_1_0, ..., emb_T_E, class_label]
        where E = embedding_length and T = number of tokens.

        :param instance: Flat tensor of length T * E + 1 where the last
                         value is the class label.
        :return: 2D tensor of shape (max_sequence_length, word_embedding_length).
        """
        embedding_length = self.__word_embedding_length
        max_length = self.__max_sequence_length

        # Total flat values minus the class label at the end
        instance_length = instance.getShape()[0] - 1

        # How many complete tokens are in this instance
        provided_tokens = instance_length // embedding_length

        values = []

        # Build the 2D matrix row by row (one row per token position)
        for i in range(max_length):
            if i < provided_tokens:
                # This token exists in the instance: copy its embedding values
                base = i * embedding_length
                for j in range(embedding_length):
                    values.append(instance.getValue((base + j,)))
            else:
                # This token position is beyond the sentence length:
                # fill with zeros (zero-padding for shorter sentences)
                for _ in range(embedding_length):
                    values.append(0.0)

        return Tensor(values, (max_length, embedding_length))

    def createClassLabelTensor(self, class_label: int) -> Tensor:
        """
        Creates a one-hot row tensor for the given class label.

        Example: class_label=2, class_label_size=4
            Output: [0.0, 0.0, 1.0, 0.0]

        :param class_label: Integer class label.
        :return: One-hot tensor of shape (1, class_label_size).
        """
        class_label_size = self.parameters.getClassLabelSize()

        # Build one-hot vector: 1.0 at the class_label position, 0.0 elsewhere
        values = [1.0 if i == class_label else 0.0 for i in range(class_label_size)]
        return Tensor(values, (1, class_label_size))

    def extractClassLabel(self, instance: Tensor) -> int:
        """
        Extracts the integer class label from the last position of the
        flat instance tensor.

        :param instance: Flat instance tensor.
        :return: Integer class label.
        """
        # The class label is always the very last value in the flat tensor
        last_index = instance.getShape()[0] - 1
        return int(instance.getValue((last_index,)))

    def train(self, train_set: List[Tensor]) -> None:
        """
        Builds the full computational graph and trains the CNN model.

        The graph structure follows the AlexNet-NLP diagram:

        Convolutional block (Conv1 through Conv5):
            conv layer 0 = Conv1 (kernel=5, 96 filters, pool after)
            conv layer 1 = Conv2 (kernel=5, 256 filters, pool after)
            conv layer 2 = Conv3 (kernel=3, 384 filters, no pool)
            conv layer 3 = Conv4 (kernel=3, 384 filters, no pool)
            conv layer 4 = Conv5 (kernel=3, 256 filters, pool after)

        Each conv layer is Im2Col1D(is_biased=True) -> MultiplicationNode(W)
        -> activation, optionally followed by MaxPool1D.
        The is_biased flag on Im2Col1D appends a bias column, turning the
        output shape from (L_out, K*C_in) into (L_out, K*C_in + 1).
        The conv weight matrix absorbs the bias as its last row.

        Fully connected block with Dropout:
            fc_hidden_layers[0] = FC6 (4096 units)
            fc_hidden_layers[1] = FC7 (4096 units)

        Each FC hidden layer is: MultiplicationNode(W) -> activation -> Dropout.
        Dropout is applied between FC hidden layers to prevent overfitting,
        matching standard AlexNet practice. The dropout ratio comes from the
        parameter object (default 0.0 means no dropout).

        Final classification head (FC8 in the AlexNet-NLP diagram):
            dense layer mapping to class_label_size outputs, followed by Softmax.

        :param train_set: Training set of flat instance tensors.
        """
        # Initialize random generator with seed for reproducibility
        random_generator = random.Random(self.parameters.getSeed())
        params: ConvolutionalNeuralNetworkParameter = self.parameters

        # =====================================================================
        # PART 1: BUILD THE COMPUTATIONAL GRAPH
        # =====================================================================

        # Create the input node. This will hold the 2D input tensor
        # (max_sequence_length, word_embedding_length) during training.
        # learnable=False because this is input data, not a weight.
        # is_biased=False because we don't add bias to the raw input.
        input_node = MultiplicationNode(False, False)
        self.input_nodes.append(input_node)

        # 'current' tracks the latest node in the graph as we build it.
        # 'current_length' and 'current_channels' track the tensor shape
        # flowing through the graph so we know the correct weight dimensions.
        current = input_node
        current_length = self.__max_sequence_length
        current_channels = self.__word_embedding_length

        # -----------------------------------------------------------------
        # CONVOLUTIONAL BLOCK: Build Conv1 through Conv5
        # -----------------------------------------------------------------
        # Loop through each convolutional layer defined in the parameters.
        # For the AlexNet-NLP diagram: i=0 is Conv1, i=1 is Conv2, etc.
        for i in range(params.convSize()):
            kernel_size = params.getConvKernelSize(i)
            stride = params.getConvStride(i)
            out_channels = params.getConvFilter(i)

            # Step 1: Im2Col1D rearranges input (L, C) into (L_out, K*C).
            # is_biased=True adds a bias column, making it (L_out, K*C + 1).
            # This allows the next weight matrix to learn bias values.
            im2col_out = self.addEdge(current, Im2Col1D(kernel_size, stride), True)

            # Update the sequence length after this convolution
            current_length = self.computeConvOutputLength(current_length, kernel_size, stride)

            # Step 2: Create a learnable weight matrix for this conv layer.
            # Shape: (K * C_in + 1, C_out) where +1 is for the bias row.
            # The framework's optimizer will update these weights during backprop.
            in_features_with_bias = kernel_size * current_channels + 1
            conv_weight = MultiplicationNode(
                Tensor(
                    params.initializeWeights(
                        in_features_with_bias, out_channels, random_generator
                    ),
                    (in_features_with_bias, out_channels),
                )
            )

            # Step 3: Matrix multiply im2col output by weights.
            # Result shape: (L_out, C_out) — this IS the convolution result.
            conv_out = self.addEdge(im2col_out, conv_weight)
            current_channels = out_channels

            # Step 4: Apply the activation function (e.g. ReLU).
            # This introduces non-linearity so the network can learn
            # complex patterns beyond simple linear combinations.
            current = self.addEdge(conv_out, params.getConvActivationFunction(i))

            # Step 5 (optional): Apply MaxPool1D if this layer has pooling.
            # Pooling reduces the sequence length and provides translation
            # invariance (small shifts in input don't change the output much).
            if params.hasPool(i):
                pool_kernel = params.getPoolKernelSize(i)
                pool_stride = params.getPoolStride(i)
                current = self.addEdge(current, MaxPool1D(pool_kernel, pool_stride))

                # Update the sequence length after pooling
                current_length = self.computeConvOutputLength(
                    current_length, pool_kernel, pool_stride
                )

        # -----------------------------------------------------------------
        # FLATTEN: Bridge between conv block and FC block
        # -----------------------------------------------------------------
        # Convert the 2D conv output (L, C) into a 1D row vector (1, L*C).
        # is_biased=True appends a bias column for the first FC layer.
        current = self.addEdge(current, Flatten(), True)
        flatten_features = current_length * current_channels

        # -----------------------------------------------------------------
        # FULLY CONNECTED BLOCK: Build FC6 and FC7 (with Dropout)
        # -----------------------------------------------------------------
        # prev_features_with_bias tracks the input size for each FC layer.
        # +1 accounts for the bias column added by the previous layer.
        prev_features_with_bias = flatten_features + 1
        dropout_ratio = params.getDropout()

        # Loop through each FC hidden layer.
        # For AlexNet-NLP: i=0 is FC6 (4096), i=1 is FC7 (4096).
        for i in range(params.fcSize()):
            hidden_size = params.getFcHiddenLayer(i)

            # Create a learnable weight matrix for this FC layer.
            # Shape: (prev_features + 1, hidden_size)
            fc_weight = MultiplicationNode(
                Tensor(
                    params.initializeWeights(
                        prev_features_with_bias, hidden_size, random_generator
                    ),
                    (prev_features_with_bias, hidden_size),
                )
            )

            # Matrix multiply: input * weights = FC output
            fc_out = self.addEdge(current, fc_weight)

            # Apply activation and optionally Dropout.
            # If Dropout is active: activation is NOT biased (Dropout adds
            # bias instead). This avoids adding bias twice.
            # If no Dropout: activation itself adds the bias column.
            if dropout_ratio > 0.0:
                # Activation without bias (Dropout will add bias)
                current = self.addEdge(fc_out, params.getFcActivationFunction(i), False)

                # Dropout randomly sets some values to zero during training
                # to prevent overfitting. is_biased=True adds bias column.
                current = self.addEdge(
                    current,
                    Dropout(dropout_ratio, random.Random(params.getSeed() + i)),
                    True
                )
            else:
                # No dropout: activation adds the bias column directly
                current = self.addEdge(fc_out, params.getFcActivationFunction(i), True)

            # Update the input size for the next FC layer
            prev_features_with_bias = hidden_size + 1

        # -----------------------------------------------------------------
        # FINAL CLASSIFICATION HEAD: FC8 (output layer)
        # -----------------------------------------------------------------
        # The last FC layer maps to class_label_size outputs (one per class).
        # No activation here — Softmax is applied separately.
        class_label_size = params.getClassLabelSize()
        output_weight = MultiplicationNode(
            Tensor(
                params.initializeWeights(
                    prev_features_with_bias, class_label_size, random_generator
                ),
                (prev_features_with_bias, class_label_size),
            )
        )

        # Compute logits (raw scores before softmax)
        logits = self.addEdge(current, output_weight)

        # Apply Softmax to convert logits into probabilities.
        # Each output value will be between 0 and 1, summing to 1.
        self.output_node = self.addEdge(logits, Softmax())

        # -----------------------------------------------------------------
        # LOSS FUNCTION: Connect output and gold label for training
        # -----------------------------------------------------------------
        # Create a node to hold the correct (gold) class label.
        # During training, we set this to a one-hot vector.
        class_label_node = ComputationalNode()
        self.input_nodes.append(class_label_node)

        # Connect the softmax output and the gold label to the loss function
        # (CrossEntropyLoss). This is what backpropagation uses to compute
        # "how wrong was the prediction" and calculate gradients.
        loss_inputs = [self.output_node, class_label_node]
        self.addFunctionEdge(loss_inputs, params.getLossFunction(), False)

        # =====================================================================
        # PART 2: TRAINING LOOP
        # =====================================================================

        # Repeat for the specified number of epochs
        for _ in range(params.getEpoch()):

            # Shuffle the training set randomly at the start of each epoch.
            # This prevents the model from memorizing the order of examples.
            for _ in range(len(train_set)):
                i1 = random_generator.randint(0, len(train_set) - 1)
                i2 = random_generator.randint(0, len(train_set) - 1)
                train_set[i1], train_set[i2] = train_set[i2], train_set[i1]

            # Process each training instance one by one (batch size = 1)
            for instance in train_set:
                # Set the input: reshape flat instance into 2D (L, E) tensor
                input_node.setValue(self.createInputTensor(instance))

                # Set the gold label: extract class label and convert to one-hot
                class_label = self.extractClassLabel(instance)
                class_label_node.setValue(self.createClassLabelTensor(class_label))

                # Forward pass: compute prediction through the entire graph
                self.forwardCalculation()

                # Backward pass: compute gradients and update all weights
                self.backpropagation()

            # After each epoch, adjust the learning rate.
            # Some optimizers decrease the learning rate over time
            # to make smaller, more precise updates as training progresses.
            params.getOptimizer().setLearningRate()

    def getOutputValue(self, output_node: ComputationalNode) -> List[float]:
        """
        Extracts the predicted class index from the softmax output.

        Finds the argmax (index of the highest probability) across
        all class positions in the softmax output row.

        :param output_node: Output computational node holding softmax probabilities.
        :return: Single-element list with the argmax class index as a float.
        """
        output_value = output_node.getValue()
        if output_value is None:
            return []

        shape = output_value.getShape()
        cols = shape[1]

        # Find the class index with the highest softmax probability
        max_val = float("-inf")
        max_index = -1

        # Iterate over each class position in the softmax output row
        for j in range(cols):
            val = output_value.getValue((0, j))
            if val > max_val:
                max_val = val
                max_index = j

        return [float(max_index)]

    def test(self, test_set: List[Tensor]) -> float:
        """
        Evaluates the model on the test set and returns accuracy.

        For each test instance: sets the input, runs forward pass (predict),
        compares the predicted class to the true class label.

        :param test_set: Test set of flat instance tensors.
        :return: Accuracy as a float in [0, 1].
        """
        input_node = self.input_nodes[0]
        count = 0
        total = 0

        # Evaluate each test instance one by one
        for instance in test_set:
            # Set the input tensor (same reshaping as in training)
            input_node.setValue(self.createInputTensor(instance))

            # Run forward pass only (no backprop) and get the predicted class
            predicted = int(self.predict()[0])

            # Compare prediction to the true label and count correct ones
            if predicted == self.extractClassLabel(instance):
                count += 1
            total += 1

        # Return accuracy as correct / total
        if total == 0:
            return 0.0
        return count / total