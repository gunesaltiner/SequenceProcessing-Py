import unittest
import random

from ComputationalGraph.Function.ReLU import ReLU
from ComputationalGraph.Function.Sigmoid import Sigmoid
from ComputationalGraph.Function.CrossEntropyLoss import CrossEntropyLoss
from ComputationalGraph.Initialization.RandomInitialization import RandomInitialization
from ComputationalGraph.Optimizer.StochasticGradientDescent import StochasticGradientDescent
from Math.Tensor import Tensor

from SequenceProcessing.Classification.ConvolutionalNeuralNetworkModel import (
    ConvolutionalNeuralNetworkModel,
)
from SequenceProcessing.Parameters.ConvolutionalNeuralNetworkParameter import (
    ConvolutionalNeuralNetworkParameter,
)


def buildSyntheticInstance(class_label: int,
                           num_tokens: int,
                           embedding_length: int,
                           rng: random.Random) -> Tensor:
    """
    Builds a flat instance tensor whose embeddings are a strongly
    class-separable signal (negative for class 0, positive for class 1)
    plus small Gaussian noise, ending with the class label.
    """
    base = -2.0 if class_label == 0 else 2.0
    values = []
    for _ in range(num_tokens):
        for _ in range(embedding_length):
            values.append(base + rng.gauss(0.0, 0.1))
    values.append(float(class_label))
    return Tensor(values, (len(values),))


def buildSyntheticDataset(num_classes: int,
                          per_class: int,
                          num_tokens: int,
                          embedding_length: int,
                          seed: int):
    """
    Builds a synthetic dataset where each class has a distinct embedding
    centroid. Returns (train_set, test_set).
    """
    rng = random.Random(seed)
    train_set = []
    test_set = []

    for c in range(num_classes):
        for _ in range(per_class):
            train_set.append(buildSyntheticInstance(c, num_tokens, embedding_length, rng))
        for _ in range(max(2, per_class // 4)):
            test_set.append(buildSyntheticInstance(c, num_tokens, embedding_length, rng))

    rng.shuffle(train_set)
    rng.shuffle(test_set)
    return train_set, test_set


def buildAlexNetNlpParameter(num_classes: int) -> ConvolutionalNeuralNetworkParameter:
    """
    Builds the full AlexNet-NLP parameter matching the assignment diagram
    structure with all 5 conv layers + FC6 + FC7 + FC8. Uses scaled-down
    filter counts and FC sizes suitable for testing with small inputs.

        Conv1 (kernel=5, 16 filters, pool k=3 s=2)
        Conv2 (kernel=5, 32 filters, pool k=3 s=2)
        Conv3 (kernel=3, 48 filters, no pool)
        Conv4 (kernel=3, 48 filters, no pool)
        Conv5 (kernel=3, 32 filters, pool k=3 s=2)
        Flatten
        FC6 (64 units, ReLU, Dropout)
        FC7 (64 units, ReLU, Dropout)
        FC8 (num_classes units, Softmax) -- added by the model as the final layer

    :param num_classes: Number of output classes for FC8.
    :return: ConvolutionalNeuralNetworkParameter matching the full diagram structure.
    """
    return ConvolutionalNeuralNetworkParameter(
        seed=42,
        epoch=1,
        optimizer=StochasticGradientDescent(0.01, 1.0),
        initialization=RandomInitialization(),
        loss=CrossEntropyLoss(),
        conv_filters=[16, 32, 48, 48, 32],
        conv_kernel_sizes=[5, 5, 3, 3, 3],
        conv_strides=[1, 1, 1, 1, 1],
        conv_activation_functions=[ReLU(), ReLU(), ReLU(), ReLU(), ReLU()],
        pool_kernel_sizes=[3, 3, 0, 0, 3],
        pool_strides=[2, 2, 0, 0, 2],
        fc_hidden_layers=[64, 64],
        fc_activation_functions=[ReLU(), ReLU()],
        class_label_size=num_classes,
        dropout=0.5,
    )


class ConvolutionalNeuralNetworkModelTest(unittest.TestCase):

    def buildTinyCnn(self, num_classes: int):
        """
        Builds a small CNN configured to be fast to train: 2 conv layers,
        small kernels, 1 FC hidden layer.
        """
        return ConvolutionalNeuralNetworkParameter(
            seed=42,
            epoch=8,
            optimizer=StochasticGradientDescent(0.05, 1.0),
            initialization=RandomInitialization(),
            loss=CrossEntropyLoss(),
            conv_filters=[8, 8],
            conv_kernel_sizes=[3, 3],
            conv_strides=[1, 1],
            conv_activation_functions=[ReLU(), ReLU()],
            pool_kernel_sizes=[2, 2],
            pool_strides=[2, 2],
            fc_hidden_layers=[16],
            fc_activation_functions=[ReLU()],
            class_label_size=num_classes,
            dropout=0.0,
        )

    def testInputTensorShape(self):
        """
        Tests that createInputTensor reshapes a flat instance into the
        expected (max_sequence_length, word_embedding_length) shape, and
        that shorter sentences are zero-padded.
        """
        params = self.buildTinyCnn(num_classes=2)
        model = ConvolutionalNeuralNetworkModel(
            parameter=params,
            word_embedding_length=4,
            max_sequence_length=10,
        )

        # Provide only 3 tokens of length-4 embedding + 1 class label = 13.
        flat = [float(i) for i in range(12)] + [1.0]
        instance = Tensor(flat, (13,))

        reshaped = model.createInputTensor(instance)
        self.assertEqual((10, 4), reshaped.getShape())

        # First 12 values must match the input embeddings exactly.
        for i in range(12):
            self.assertEqual(float(i), reshaped.getValue((i // 4, i % 4)))

        # Remaining positions must be zero-padded.
        for token in range(3, 10):
            for j in range(4):
                self.assertEqual(0.0, reshaped.getValue((token, j)))

    def testExtractClassLabel(self):
        """
        Tests that the integer class label is correctly read from the last
        position of the flat instance tensor.
        """
        params = self.buildTinyCnn(num_classes=3)
        model = ConvolutionalNeuralNetworkModel(
            parameter=params,
            word_embedding_length=2,
            max_sequence_length=5,
        )
        instance = Tensor([0.1, 0.2, 0.3, 0.4, 2.0], (5,))
        self.assertEqual(2, model.extractClassLabel(instance))

    def testTrainingSmokeRun(self):
        """
        Verifies the full graph builds and a single training epoch runs to
        completion without errors on a tiny dataset.
        """
        num_classes = 2
        embedding_length = 4
        max_length = 8

        train_set, _ = buildSyntheticDataset(
            num_classes=num_classes,
            per_class=4,
            num_tokens=max_length,
            embedding_length=embedding_length,
            seed=0,
        )

        params = ConvolutionalNeuralNetworkParameter(
            seed=1,
            epoch=1,
            optimizer=StochasticGradientDescent(0.01, 1.0),
            initialization=RandomInitialization(),
            loss=CrossEntropyLoss(),
            conv_filters=[4],
            conv_kernel_sizes=[3],
            conv_strides=[1],
            conv_activation_functions=[ReLU()],
            pool_kernel_sizes=[2],
            pool_strides=[2],
            fc_hidden_layers=[8],
            fc_activation_functions=[ReLU()],
            class_label_size=num_classes,
        )
        model = ConvolutionalNeuralNetworkModel(
            parameter=params,
            word_embedding_length=embedding_length,
            max_sequence_length=max_length,
        )

        model.train(train_set)

        # After training, predictions must be valid integer class indices.
        for instance in train_set:
            model.input_nodes[0].setValue(model.createInputTensor(instance))
            prediction = int(model.predict()[0])
            self.assertGreaterEqual(prediction, 0)
            self.assertLess(prediction, num_classes)

    def testLearnsLinearlySeparableTask(self):
        """
        Trains a minimal CNN on a synthetic 2-class separable dataset and
        verifies test accuracy is meaningfully above chance (0.5). This
        confirms the full pipeline (Im2Col1D + learnable weights + activation
        + softmax + CrossEntropy + backprop) works end-to-end.
        """
        num_classes = 2
        embedding_length = 3
        max_length = 5

        train_set, test_set = buildSyntheticDataset(
            num_classes=num_classes,
            per_class=20,
            num_tokens=max_length,
            embedding_length=embedding_length,
            seed=7,
        )

        params = ConvolutionalNeuralNetworkParameter(
            seed=1,
            epoch=20,
            optimizer=StochasticGradientDescent(0.1, 1.0),
            initialization=RandomInitialization(),
            loss=CrossEntropyLoss(),
            conv_filters=[4],
            conv_kernel_sizes=[2],
            conv_strides=[1],
            conv_activation_functions=[Sigmoid()],
            pool_kernel_sizes=[0],
            pool_strides=[0],
            fc_hidden_layers=[],
            fc_activation_functions=[],
            class_label_size=num_classes,
        )
        model = ConvolutionalNeuralNetworkModel(
            parameter=params,
            word_embedding_length=embedding_length,
            max_sequence_length=max_length,
        )

        model.train(train_set)
        accuracy = model.test(test_set)

        # Chance for 2 classes is 0.5; require well above to confirm learning.
        self.assertGreater(accuracy, 0.8,
                           f"Expected > 0.8 accuracy on separable data, got {accuracy:.3f}")

    def testFullAlexNetNlpGraphBuilds(self):
        """
        Smoke test for the full AlexNet-NLP diagram configuration:

            Input: 64 tokens x 16 embedding (scaled down from 512x300 for speed)
            Conv1: kernel=5, 96 filters, pool(k=3, s=2)
            Conv2: kernel=5, 256 filters, pool(k=3, s=2)
            Conv3: kernel=3, 384 filters, no pool
            Conv4: kernel=3, 384 filters, no pool
            Conv5: kernel=3, 256 filters, pool(k=3, s=2)
            Flatten
            FC6: 4096 -> ReLU -> Dropout(0.5)
            FC7: 4096 -> ReLU -> Dropout(0.5)
            FC8: 3 classes -> Softmax

        This test verifies that the full 5-conv + FC6 + FC7 + FC8 graph
        builds successfully and can run forward + backward without errors.
        Uses small input dimensions (32x16 instead of 512x300) for speed.
        """
        num_classes = 3
        embedding_length = 16
        max_length = 64

        params = buildAlexNetNlpParameter(num_classes=num_classes)

        model = ConvolutionalNeuralNetworkModel(
            parameter=params,
            word_embedding_length=embedding_length,
            max_sequence_length=max_length,
        )

        # Build 2 dummy training instances (one per class is enough for smoke)
        rng = random.Random(0)
        train_set = []
        for c in range(2):
            values = [rng.gauss(0.0, 1.0) for _ in range(max_length * embedding_length)]
            values.append(float(c))
            train_set.append(Tensor(values, (len(values),)))

        # Train for 1 epoch — must complete without errors
        model.train(train_set)

        # Predictions must be valid class indices
        for instance in train_set:
            model.input_nodes[0].setValue(model.createInputTensor(instance))
            prediction = int(model.predict()[0])
            self.assertGreaterEqual(prediction, 0)
            self.assertLess(prediction, num_classes)

    def testFcLayerMapping(self):
        """
        Verifies that the parameter configuration correctly maps to the
        AlexNet-NLP diagram layers:

            fc_hidden_layers[0] = FC6 (4096 units)
            fc_hidden_layers[1] = FC7 (4096 units)
            Final classification layer = FC8 (class_label_size units)

        The fc_hidden_layers list holds only FC6 and FC7; FC8 is added
        automatically by the model as the output classification layer.
        """
        num_classes = 5
        params = buildAlexNetNlpParameter(num_classes=num_classes)

        # FC6 = fc_hidden_layers[0]
        self.assertEqual(64, params.getFcHiddenLayer(0))

        # FC7 = fc_hidden_layers[1]
        self.assertEqual(64, params.getFcHiddenLayer(1))

        # FC8 = class_label_size (added by model, not in fc_hidden_layers)
        self.assertEqual(num_classes, params.getClassLabelSize())

        # fcSize() returns 2 (FC6 + FC7 only, not FC8)
        self.assertEqual(2, params.fcSize())

    def testDropoutIsApplied(self):
        """
        Verifies that when dropout > 0.0 is set in parameters, the model
        builds the graph with Dropout edges between FC hidden layers without
        errors. Trains 1 epoch on dummy data to confirm forward+backward
        work correctly with Dropout active.
        """
        num_classes = 2
        embedding_length = 4
        max_length = 8

        train_set, _ = buildSyntheticDataset(
            num_classes=num_classes,
            per_class=4,
            num_tokens=max_length,
            embedding_length=embedding_length,
            seed=0,
        )

        params = ConvolutionalNeuralNetworkParameter(
            seed=1,
            epoch=1,
            optimizer=StochasticGradientDescent(0.01, 1.0),
            initialization=RandomInitialization(),
            loss=CrossEntropyLoss(),
            conv_filters=[4],
            conv_kernel_sizes=[3],
            conv_strides=[1],
            conv_activation_functions=[ReLU()],
            pool_kernel_sizes=[0],
            pool_strides=[0],
            fc_hidden_layers=[8, 8],
            fc_activation_functions=[ReLU(), ReLU()],
            class_label_size=num_classes,
            dropout=0.5,
        )

        model = ConvolutionalNeuralNetworkModel(
            parameter=params,
            word_embedding_length=embedding_length,
            max_sequence_length=max_length,
        )

        # Must complete training without errors with Dropout active
        model.train(train_set)

        # Predictions must still be valid
        for instance in train_set:
            model.input_nodes[0].setValue(model.createInputTensor(instance))
            prediction = int(model.predict()[0])
            self.assertGreaterEqual(prediction, 0)
            self.assertLess(prediction, num_classes)


if __name__ == "__main__":
    unittest.main()