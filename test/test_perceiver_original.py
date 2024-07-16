import unittest

import torch
import torch.nn as nn

from src.perceiver_core.perceiver_original import Perceiver


class TestPerceiver(unittest.TestCase):
    def setUp(self):
        """Set up a basic Perceiver model for testing."""
        self.model = Perceiver(
            num_fourier_bands=4,
            num_layers=2,
            max_frequency=10.0,
            input_channels=3,
            num_input_axes=2,
            num_latent_tokens=32,
            latent_dimension=64,
            cross_attention_heads=1,
            self_attention_heads=2,
            cross_attention_head_dim=32,
            self_attention_head_dim=32,
            num_classes=10,
            attention_dropout=0.1,
            feedforward_dropout=0.1,
            weight_tie_layers=False,
            use_fourier_encoding=True,
            self_attentions_per_cross=1,
            include_classifier_head=True,
        )

    def test_model_initialization(self):
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model, nn.Module)
        # check if some attributes are set correctly
        self.assertEqual(self.model.num_input_axes, 2)
        self.assertEqual(self.model.latent_dimension, 64)
        self.assertEqual(self.model.latent_tokens.shape, (32, 64))

    def test_forward_pass_shape(self):
        """Test if the forward pass produces the expected output shape."""
        batch_size = 2
        dimensions = (224, 224)
        input_data = torch.randn(batch_size, *dimensions, 3)

        output = self.model(input_data)
        self.assertEqual(output.shape, (batch_size, 10))

    def test_embedding_output(self):
        """Test if the model can return embeddings."""
        batch_size = 2
        dimensions = (224, 224)
        input_data = torch.randn(batch_size, *dimensions, 3)

        embeddings = self.model(input_data, return_embeddings=True)
        self.assertEqual(embeddings.shape, (batch_size, 32, 64))

    def test_fourier_encoding(self):
        """Test if Fourier encoding changes the input shape as expected."""
        batch_size = 2
        dimensions = (224, 224)
        input_data = torch.randn(batch_size, *dimensions, 3)

        encoded_input = self.model._apply_fourier_encode(input_data)
        expected_channels = 3 + (2 * 4 + 1) * 2  # input_channels + (2 * num_bands + 1) * num_axes  (variables set in self.model)

        self.assertEqual(encoded_input.shape, (batch_size, 224, 224, expected_channels))

    def test_attention_mask(self):
        """Test if the model accepts an attention mask without errors."""
        batch_size = 2
        dimensions = (224, 224)
        input_data = torch.randn(batch_size, *dimensions, 3)
        attention_mask = torch.ones(batch_size, *dimensions).bool()

        try:
            output = self.model(input_data, attention_mask=attention_mask)
            self.assertEqual(output.shape, (batch_size, 10))
        except Exception as e:
            self.fail(f"Forward pass with attention mask raised an exception: {e}")

    def test_no_fourier_encoding(self):
        """Test if the model works without Fourier encoding."""
        model_no_fourier = Perceiver(
            num_fourier_bands=4,
            num_layers=2,
            max_frequency=10.0,
            input_channels=3,
            num_input_axes=2,
            num_latent_tokens=32,
            latent_dimension=64,
            num_classes=10,
            use_fourier_encoding=False,
        )
        batch_size = 2
        dimensions = (224, 224)
        input_data = torch.randn(batch_size, *dimensions, 3)

        output = model_no_fourier(input_data)
        self.assertEqual(output.shape, (batch_size, 10))

    def test_weight_tying(self):
        """Test if weight tying correctly shares parameters across depths."""
        model_tied = Perceiver(
            num_fourier_bands=4,
            num_layers=3,  # use at least 3 layers to test tying
            max_frequency=10.0,
            input_channels=3,
            num_input_axes=2,
            num_latent_tokens=32,
            latent_dimension=64,
            cross_attention_heads=1,
            self_attention_heads=2,
            cross_attention_head_dim=32,
            self_attention_head_dim=32,
            num_classes=10,
            attention_dropout=0.1,
            feedforward_dropout=0.1,
            weight_tie_layers=True,
        )

        def params_equal(params1, params2):
            return all(torch.equal(p1, p2) for p1, p2 in zip(params1, params2))

        # check if parameters are shared across depths (starting from depth 1)
        for i in range(
            2, len(model_tied.layers)
        ):  # each layer contains [0] cross-attention, [1] cross-feedforward, [2] list of self-attention and self-feedforward
            # check cross-attention
            print(f"Model tied layers: {model_tied.layers[i][0].parameters()}")
            self.assertTrue(
                params_equal(model_tied.layers[i][0].parameters(), model_tied.layers[1][0].parameters()),
                f"Cross-attention parameters at depth {i} should be the same as depth 1",
            )

            # check cross-feedforward
            self.assertTrue(
                params_equal(model_tied.layers[i][1].parameters(), model_tied.layers[1][1].parameters()),
                f"Cross-feedforward parameters at depth {i} should be the same as depth 1",
            )

            # check self-attention and self-feedforward
            for j in range(len(model_tied.layers[i][2])):
                self.assertTrue(
                    params_equal(
                        model_tied.layers[i][2][j][0].parameters(), model_tied.layers[1][2][j][0].parameters()
                    ),  # [2][j][0] is the j-th self-attention block, and [2][j][1] is its corrsponding ff network
                    f"Self-attention parameters at depth {i}, block {j} should be the same as depth 1",
                )
                self.assertTrue(
                    params_equal(model_tied.layers[i][2][j][1].parameters(), model_tied.layers[1][2][j][1].parameters()),
                    f"Self-feedforward parameters at depth {i}, block {j} should be the same as depth 1",
                )

        # check that parameters at depth 0 are different
        self.assertFalse(
            params_equal(model_tied.layers[0][0].parameters(), model_tied.layers[1][0].parameters()),
            "Cross-attention parameters at depth 0 should be different from other depths",
        )

        # ensure that cross-attention and self-attention parameters are different
        self.assertFalse(
            params_equal(model_tied.layers[1][0].parameters(), model_tied.layers[1][2][0][0].parameters()),
            "Cross-attention and self-attention parameters should be different",
        )

    def test_different_input_sizes(self):
        """Test if the model can handle different input sizes."""
        batch_size = 2
        for size in [32, 64, 128]:
            input_data = torch.randn(batch_size, size, size, 3)
            try:
                output = self.model(input_data)
                self.assertEqual(output.shape, (batch_size, 10))
            except Exception as e:
                self.fail(f"Failed on input size {size}: {e}")

    def test_gpu_compatibility(self):
        """Test if the model can be moved to GPU (if available)."""
        if torch.cuda.is_available():
            try:
                model_gpu = self.model.cuda()
                batch_size = 2
                dimensions = (224, 224)
                input_data = torch.randn(batch_size, *dimensions, 3).cuda()

                output = model_gpu(input_data)
                self.assertEqual(output.shape, (batch_size, 10))
                self.assertTrue(output.is_cuda)
            except Exception as e:
                self.fail(f"GPU compatibility test failed: {str(e)}")
        else:
            print("CUDA not available, skipping GPU test")


if __name__ == "__main__":
    unittest.main()
