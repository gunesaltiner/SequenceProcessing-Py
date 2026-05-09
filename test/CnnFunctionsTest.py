import unittest

from Math.Tensor import Tensor

from SequenceProcessing.Functions.Flatten import Flatten
from SequenceProcessing.Functions.MaxPool1D import MaxPool1D
from SequenceProcessing.Functions.Im2Col1D import Im2Col1D


class FlattenTest(unittest.TestCase):

    def testCalculate(self):
        """
        Tests forward flattening of a (3, 2) tensor into (1, 6).
        """
        tensor = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2))
        func = Flatten()

        result = func.calculate(tensor)

        self.assertEqual([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], result.getData())
        self.assertEqual((1, 6), result.getShape())

    def testDerivative(self):
        """
        Tests that derivative reshapes the gradient back to the input shape.
        """
        tensor = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2))
        func = Flatten()
        func.calculate(tensor)

        backward = Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], (1, 6))
        result = func.derivative(func.calculate(tensor), backward)

        self.assertEqual([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], result.getData())
        self.assertEqual((3, 2), result.getShape())


class MaxPool1DTest(unittest.TestCase):

    def testCalculate(self):
        """
        Tests forward max pooling on a (4, 2) input with kernel=2, stride=2.
        Expected output shape (2, 2) where each row holds the per-channel max
        of two consecutive input rows.
        """
        # Input rows: [1,2], [3,4], [2,5], [6,1]
        # Window 0 (rows 0-1): max per channel = [3, 4]
        # Window 1 (rows 2-3): max per channel = [6, 5]
        tensor = Tensor([1.0, 2.0, 3.0, 4.0, 2.0, 5.0, 6.0, 1.0], (4, 2))
        func = MaxPool1D(kernel_size=2, stride=2)

        result = func.calculate(tensor)

        self.assertEqual((2, 2), result.getShape())
        self.assertEqual([3.0, 4.0, 6.0, 5.0], result.getData())

    def testDerivative(self):
        """
        Tests that gradient is routed only to the argmax positions.
        Argmaxes for the example above:
          window 0 channel 0 -> row 1 (value 3)
          window 0 channel 1 -> row 1 (value 4)
          window 1 channel 0 -> row 3 (value 6)
          window 1 channel 1 -> row 2 (value 5)
        Backward gradient: [10, 20, 30, 40] (shape (2, 2))
        Expected input gradient (shape (4, 2)):
          row 0: [0, 0]   row 1: [10, 20]   row 2: [0, 40]   row 3: [30, 0]
        """
        tensor = Tensor([1.0, 2.0, 3.0, 4.0, 2.0, 5.0, 6.0, 1.0], (4, 2))
        func = MaxPool1D(kernel_size=2, stride=2)
        forward = func.calculate(tensor)

        backward = Tensor([10.0, 20.0, 30.0, 40.0], (2, 2))
        grad = func.derivative(forward, backward)

        self.assertEqual((4, 2), grad.getShape())
        self.assertEqual([0.0, 0.0, 10.0, 20.0, 0.0, 40.0, 30.0, 0.0], grad.getData())

    def testOverlappingWindowsAccumulate(self):
        """
        Tests that overlapping windows correctly accumulate gradient on
        the same input position when it is argmax in multiple windows.
        Input (5, 1), kernel=3, stride=1. Position 2 (value 9) is the
        argmax in all three windows.
        """
        tensor = Tensor([1.0, 2.0, 9.0, 3.0, 4.0], (5, 1))
        func = MaxPool1D(kernel_size=3, stride=1)
        forward = func.calculate(tensor)

        # All three output positions equal 9.
        self.assertEqual([9.0, 9.0, 9.0], forward.getData())

        backward = Tensor([1.0, 1.0, 1.0], (3, 1))
        grad = func.derivative(forward, backward)

        self.assertEqual((5, 1), grad.getShape())
        # Position 2 should accumulate gradient from all 3 windows.
        self.assertEqual([0.0, 0.0, 3.0, 0.0, 0.0], grad.getData())


class Im2Col1DTest(unittest.TestCase):

    def testCalculate(self):
        """
        Tests forward im2col on a (4, 2) input with kernel=3, stride=1.
        Expected output shape (2, 6) where each row is the flattened
        concatenation of three consecutive input rows.
        """
        # Input rows: [1,2], [3,4], [5,6], [7,8]
        # Window 0: [1,2, 3,4, 5,6]
        # Window 1: [3,4, 5,6, 7,8]
        tensor = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (4, 2))
        func = Im2Col1D(kernel_size=3, stride=1)

        result = func.calculate(tensor)

        self.assertEqual((2, 6), result.getShape())
        self.assertEqual(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
             3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            result.getData()
        )

    def testDerivativeOverlap(self):
        """
        Tests that the derivative correctly sums gradients from overlapping
        windows. With kernel=3 stride=1 and input length 4, input row 2
        appears in both windows and so its gradient should accumulate.
        Backward = ones((2, 6)), expected input grad:
          row 0: appears in 1 window -> [1, 1]
          row 1: appears in 2 windows -> [2, 2]
          row 2: appears in 2 windows -> [2, 2]
          row 3: appears in 1 window -> [1, 1]
        """
        tensor = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (4, 2))
        func = Im2Col1D(kernel_size=3, stride=1)
        forward = func.calculate(tensor)

        backward = Tensor([1.0] * 12, (2, 6))
        grad = func.derivative(forward, backward)

        self.assertEqual((4, 2), grad.getShape())
        self.assertEqual([1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0], grad.getData())

    def testStridedNoOverlap(self):
        """
        Tests im2col with stride equal to kernel size (no overlap).
        Input (6, 1), kernel=3, stride=3 -> output (2, 3).
        """
        tensor = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (6, 1))
        func = Im2Col1D(kernel_size=3, stride=3)

        result = func.calculate(tensor)

        self.assertEqual((2, 3), result.getShape())
        self.assertEqual([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], result.getData())

        # Each input row appears in exactly one window, so backward of
        # ones should map to all-ones input gradient.
        backward = Tensor([1.0] * 6, (2, 3))
        grad = func.derivative(result, backward)
        self.assertEqual([1.0] * 6, grad.getData())


if __name__ == "__main__":
    unittest.main()