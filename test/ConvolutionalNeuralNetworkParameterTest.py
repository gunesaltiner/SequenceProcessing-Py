import unittest

from SequenceProcessing.Parameters.ConvolutionalNeuralNetworkParameter import (
    ConvolutionalNeuralNetworkParameter,
)


class DummyFunction:
    pass


class DummyOptimizer:
    pass


class DummyInitialization:
    pass


def buildAlexNetNlpParameter() -> ConvolutionalNeuralNetworkParameter:
    """
    Helper that builds the parameter object matching the AlexNet-NLP
    architecture from the assignment diagram (5 conv + 3 FC, with pools
    after conv1, conv2, conv5 in real AlexNet style).
    """
    return ConvolutionalNeuralNetworkParameter(
        seed=1,
        epoch=10,
        optimizer=DummyOptimizer(),
        initialization=DummyInitialization(),
        loss=DummyFunction(),
        conv_filters=[96, 256, 384, 384, 256],
        conv_kernel_sizes=[5, 5, 3, 3, 3],
        conv_strides=[1, 1, 1, 1, 1],
        conv_activation_functions=[DummyFunction() for _ in range(5)],
        pool_kernel_sizes=[3, 3, 0, 0, 3],
        pool_strides=[2, 2, 0, 0, 2],
        fc_hidden_layers=[4096, 4096],
        fc_activation_functions=[DummyFunction(), DummyFunction()],
        class_label_size=10,
        dropout=0.5,
    )


class ConvolutionalNeuralNetworkParameterTest(unittest.TestCase):

    def testBasicProperties(self):
        """
        Tests basic parameter accessor behavior on a 5-conv + 2-FC AlexNet-NLP setup.
        """
        param = buildAlexNetNlpParameter()

        self.assertEqual(5, param.convSize())
        self.assertEqual(2, param.fcSize())
        self.assertEqual(10, param.getClassLabelSize())

    def testConvAccessors(self):
        """
        Tests per-layer convolution getters.
        """
        param = buildAlexNetNlpParameter()

        self.assertEqual(96, param.getConvFilter(0))
        self.assertEqual(256, param.getConvFilter(1))
        self.assertEqual(384, param.getConvFilter(2))
        self.assertEqual(5, param.getConvKernelSize(0))
        self.assertEqual(3, param.getConvKernelSize(2))
        self.assertEqual(1, param.getConvStride(0))

    def testFcAccessors(self):
        """
        Tests fully connected layer accessors.
        """
        param = buildAlexNetNlpParameter()

        self.assertEqual(4096, param.getFcHiddenLayer(0))
        self.assertEqual(4096, param.getFcHiddenLayer(1))

    def testPoolAccessors(self):
        """
        Tests that pool kernel/stride and the hasPool helper match the diagram:
        pools after conv1, conv2, and conv5; no pool after conv3 or conv4.
        """
        param = buildAlexNetNlpParameter()

        self.assertTrue(param.hasPool(0))
        self.assertTrue(param.hasPool(1))
        self.assertFalse(param.hasPool(2))
        self.assertFalse(param.hasPool(3))
        self.assertTrue(param.hasPool(4))

        self.assertEqual(3, param.getPoolKernelSize(0))
        self.assertEqual(2, param.getPoolStride(0))
        self.assertEqual(0, param.getPoolKernelSize(2))

    def testInheritedProperties(self):
        """
        Tests that values passed up to NeuralNetworkParameter are accessible.
        """
        param = buildAlexNetNlpParameter()

        self.assertEqual(1, param.getSeed())
        self.assertEqual(10, param.getEpoch())
        self.assertEqual(0.5, param.getDropout())
        self.assertEqual(1, param.getBatchSize())

    def testListGetters(self):
        """
        Tests that list-form getters return the correct collections.
        """
        param = buildAlexNetNlpParameter()

        self.assertEqual([96, 256, 384, 384, 256], param.getConvFilters())
        self.assertEqual([5, 5, 3, 3, 3], param.getConvKernelSizes())
        self.assertEqual([1, 1, 1, 1, 1], param.getConvStrides())
        self.assertEqual([3, 3, 0, 0, 3], param.getPoolKernelSizes())
        self.assertEqual([2, 2, 0, 0, 2], param.getPoolStrides())
        self.assertEqual([4096, 4096], param.getFcHiddenLayers())

    def testMismatchedConvListsRaise(self):
        """
        Tests that mismatched conv list lengths raise ValueError.
        """
        with self.assertRaises(ValueError):
            ConvolutionalNeuralNetworkParameter(
                seed=1,
                epoch=1,
                optimizer=DummyOptimizer(),
                initialization=DummyInitialization(),
                loss=DummyFunction(),
                conv_filters=[96, 256],
                conv_kernel_sizes=[5],
                conv_strides=[1, 1],
                conv_activation_functions=[DummyFunction(), DummyFunction()],
                pool_kernel_sizes=[3, 3],
                pool_strides=[2, 2],
                fc_hidden_layers=[64],
                fc_activation_functions=[DummyFunction()],
                class_label_size=5,
            )

    def testMismatchedFcListsRaise(self):
        """
        Tests that mismatched FC list lengths raise ValueError.
        """
        with self.assertRaises(ValueError):
            ConvolutionalNeuralNetworkParameter(
                seed=1,
                epoch=1,
                optimizer=DummyOptimizer(),
                initialization=DummyInitialization(),
                loss=DummyFunction(),
                conv_filters=[96],
                conv_kernel_sizes=[5],
                conv_strides=[1],
                conv_activation_functions=[DummyFunction()],
                pool_kernel_sizes=[3],
                pool_strides=[2],
                fc_hidden_layers=[64, 32],
                fc_activation_functions=[DummyFunction()],
                class_label_size=5,
            )

    def testInvalidClassLabelSizeRaises(self):
        """
        Tests that a non-positive class_label_size raises ValueError.
        """
        with self.assertRaises(ValueError):
            ConvolutionalNeuralNetworkParameter(
                seed=1,
                epoch=1,
                optimizer=DummyOptimizer(),
                initialization=DummyInitialization(),
                loss=DummyFunction(),
                conv_filters=[96],
                conv_kernel_sizes=[5],
                conv_strides=[1],
                conv_activation_functions=[DummyFunction()],
                pool_kernel_sizes=[0],
                pool_strides=[0],
                fc_hidden_layers=[64],
                fc_activation_functions=[DummyFunction()],
                class_label_size=0,
            )

    def testNegativeKernelSizeRaises(self):
        """
        Tests that a non-positive conv kernel size raises ValueError.
        """
        with self.assertRaises(ValueError):
            ConvolutionalNeuralNetworkParameter(
                seed=1,
                epoch=1,
                optimizer=DummyOptimizer(),
                initialization=DummyInitialization(),
                loss=DummyFunction(),
                conv_filters=[96],
                conv_kernel_sizes=[0],
                conv_strides=[1],
                conv_activation_functions=[DummyFunction()],
                pool_kernel_sizes=[0],
                pool_strides=[0],
                fc_hidden_layers=[64],
                fc_activation_functions=[DummyFunction()],
                class_label_size=5,
            )

    def testNegativeFilterCountRaises(self):
        """
        Tests that a non-positive filter count raises ValueError.
        """
        with self.assertRaises(ValueError):
            ConvolutionalNeuralNetworkParameter(
                seed=1,
                epoch=1,
                optimizer=DummyOptimizer(),
                initialization=DummyInitialization(),
                loss=DummyFunction(),
                conv_filters=[-1],
                conv_kernel_sizes=[3],
                conv_strides=[1],
                conv_activation_functions=[DummyFunction()],
                pool_kernel_sizes=[0],
                pool_strides=[0],
                fc_hidden_layers=[64],
                fc_activation_functions=[DummyFunction()],
                class_label_size=5,
            )

    def testNegativeStrideRaises(self):
        """
        Tests that a non-positive stride raises ValueError.
        """
        with self.assertRaises(ValueError):
            ConvolutionalNeuralNetworkParameter(
                seed=1,
                epoch=1,
                optimizer=DummyOptimizer(),
                initialization=DummyInitialization(),
                loss=DummyFunction(),
                conv_filters=[96],
                conv_kernel_sizes=[3],
                conv_strides=[0],
                conv_activation_functions=[DummyFunction()],
                pool_kernel_sizes=[0],
                pool_strides=[0],
                fc_hidden_layers=[64],
                fc_activation_functions=[DummyFunction()],
                class_label_size=5,
            )

    def testPoolEnabledWithZeroStrideRaises(self):
        """
        Tests that enabling pooling (kernel > 0) with stride <= 0 raises ValueError.
        """
        with self.assertRaises(ValueError):
            ConvolutionalNeuralNetworkParameter(
                seed=1,
                epoch=1,
                optimizer=DummyOptimizer(),
                initialization=DummyInitialization(),
                loss=DummyFunction(),
                conv_filters=[96],
                conv_kernel_sizes=[3],
                conv_strides=[1],
                conv_activation_functions=[DummyFunction()],
                pool_kernel_sizes=[3],
                pool_strides=[0],
                fc_hidden_layers=[64],
                fc_activation_functions=[DummyFunction()],
                class_label_size=5,
            )

    def testNegativeFcHiddenLayerRaises(self):
        """
        Tests that a non-positive FC hidden layer size raises ValueError.
        """
        with self.assertRaises(ValueError):
            ConvolutionalNeuralNetworkParameter(
                seed=1,
                epoch=1,
                optimizer=DummyOptimizer(),
                initialization=DummyInitialization(),
                loss=DummyFunction(),
                conv_filters=[96],
                conv_kernel_sizes=[3],
                conv_strides=[1],
                conv_activation_functions=[DummyFunction()],
                pool_kernel_sizes=[0],
                pool_strides=[0],
                fc_hidden_layers=[0],
                fc_activation_functions=[DummyFunction()],
                class_label_size=5,
            )


if __name__ == "__main__":
    unittest.main()