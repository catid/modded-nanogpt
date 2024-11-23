# test_memory_layer.py

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from memory_layer import MemoryLayer  # Assuming the MemoryLayer is saved in memory_layer.py

class TestMemoryLayer(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        torch.manual_seed(0)
        self.batch_size = 4
        self.seq_len = 10
        self.input_dim = 512
        self.output_dim = 512
        self.K = 64
        self.tau = 8
        self.t = 1.0

        self.memory_layer = MemoryLayer(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            K=self.K,
            tau=self.tau,
            t=self.t
        )

    def test_output_shape(self):
        """Test that the output tensor has the correct shape."""
        # Create random input tensor
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)

        # Forward pass
        y_hat = self.memory_layer(x)

        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.output_dim)
        self.assertEqual(y_hat.shape, expected_shape, "Output tensor has incorrect shape.")

    def test_forward_pass(self):
        """Test that the forward pass runs without errors."""
        # Create random input tensor
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)

        # Forward pass
        try:
            y_hat = self.memory_layer(x)
        except Exception as e:
            self.fail(f"Forward pass failed with exception: {e}")

    def test_backward_pass(self):
        """Test that gradients can be computed through the MemoryLayer."""
        # Create random input tensor with requires_grad=True
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim, requires_grad=True)

        # Forward pass
        y_hat = self.memory_layer(x)

        # Compute a simple loss (mean of outputs)
        loss = y_hat.mean()

        # Backward pass
        try:
            loss.backward()
        except Exception as e:
            self.fail(f"Backward pass failed with exception: {e}")

        # Check if gradients are computed for input
        self.assertIsNotNone(x.grad, "Gradients not computed for input tensor.")
        self.assertTrue(torch.all(torch.isfinite(x.grad)), "Input gradients contain non-finite values.")

    def test_consistency(self):
        """Test that the MemoryLayer produces consistent outputs for the same input."""
        # Create random input tensor
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)

        # Forward pass twice
        y_hat1 = self.memory_layer(x)
        y_hat2 = self.memory_layer(x)

        # Check that outputs are close
        self.assertTrue(torch.allclose(y_hat1, y_hat2, atol=1e-6), "Outputs are not consistent for the same input.")

    def test_zero_input(self):
        """Test the layer's behavior with zero inputs."""
        # Create zero input tensor
        x = torch.zeros(self.batch_size, self.seq_len, self.input_dim)

        # Forward pass
        y_hat = self.memory_layer(x)

        # Check that output is finite
        self.assertTrue(torch.isfinite(y_hat).all(), "Output contains non-finite values for zero input.")

    def test_memory_layer_integration(self):
        """Test integrating MemoryLayer into a simple model."""
        # Create a simple model integrating MemoryLayer
        class SimpleModel(nn.Module):
            def __init__(self, memory_layer):
                super(SimpleModel, self).__init__()
                self.memory_layer = memory_layer
                self.output_dim = memory_layer.output_dim  # Corrected assignment
                self.linear = nn.Linear(self.output_dim, 1)

            def forward(self, x):
                y_hat = self.memory_layer(x)
                out = self.linear(y_hat)
                return out.mean()

        # Instantiate the model
        model = SimpleModel(self.memory_layer)

        # Create random input tensor
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)

        # Forward pass
        output = model(x)

        # Backward pass
        try:
            output.backward()
        except Exception as e:
            self.fail(f"Backward pass in integration test failed with exception: {e}")

    def test_large_input(self):
        """Test the MemoryLayer with larger batch size and sequence length."""
        # Create larger input tensor
        x = torch.randn(8, 50, self.input_dim)

        # Forward pass
        y_hat = self.memory_layer(x)

        # Check output shape
        expected_shape = (8, 50, self.output_dim)
        self.assertEqual(y_hat.shape, expected_shape, "Output tensor has incorrect shape for large input.")

    def test_different_parameters(self):
        """Test the MemoryLayer with different K and tau values."""
        for K, tau in [(32, 16), (128, 4), (256, 2)]:
            with self.subTest(K=K, tau=tau):
                # Initialize a new MemoryLayer
                memory_layer = MemoryLayer(
                    input_dim=self.input_dim,
                    output_dim=self.output_dim,
                    K=K,
                    tau=tau,
                    t=self.t
                )
                # Create random input tensor
                x = torch.randn(self.batch_size, self.seq_len, self.input_dim)

                # Forward pass
                y_hat = memory_layer(x)

                # Check output shape
                expected_shape = (self.batch_size, self.seq_len, self.output_dim)
                self.assertEqual(y_hat.shape, expected_shape, f"Output tensor has incorrect shape with K={K}, tau={tau}.")

    def test_gradients_through_hash_tables(self):
        """Test that gradients flow to the hash tables."""
        # Create random input tensor
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)

        # Forward pass
        y_hat = self.memory_layer(x)

        # Compute a simple loss
        loss = y_hat.mean()

        # Backward pass
        loss.backward()

        # Check if gradients are computed for hash tables
        for idx, T_k in enumerate(self.memory_layer.hash_tables):
            self.assertIsNotNone(T_k.weight.grad, f"Gradients not computed for hash table {idx}.")
            self.assertTrue(torch.all(torch.isfinite(T_k.weight.grad)), f"Hash table {idx} gradients contain non-finite values.")

if __name__ == '__main__':
    unittest.main()
