"""
Run from root folder: `python -m test.test_pos_encodings`
"""

import unittest

import torch

from bfm_model.perceiver_components.pos_encoder import (
    FourierPositionEncoding,
    PositionEncodingProjector,
    TrainablePositionEncoding,
    build_position_encoding,
)


class TestPositionEncodings(unittest.TestCase):
    def test_fourier_position_encoding(self):
        batch_size = 16
        index_dims = (8, 8)
        num_bands = 4
        max_freq = 10.0

        # Test initialization
        enc = FourierPositionEncoding(index_dims, num_bands, max_freq)
        self.assertEqual(enc.index_dims, index_dims)
        self.assertEqual(enc.num_bands, num_bands)
        self.assertEqual(enc.max_freq, max_freq)

        # Test forward pass
        output = enc(batch_size)
        expected_output_size = batch_size, *index_dims, num_bands * len(index_dims) * 2 + len(index_dims)
        self.assertEqual(output.shape, expected_output_size)

        # Test sine_only option
        enc_sine_only = FourierPositionEncoding(index_dims, num_bands, max_freq, sine_only=True)
        output_sine_only = enc_sine_only(batch_size)
        expected_output_size_sine_only = batch_size, *index_dims, num_bands * len(index_dims) + len(index_dims)
        self.assertEqual(output_sine_only.shape, expected_output_size_sine_only)

        # Test without concat_pos
        enc_no_concat = FourierPositionEncoding(index_dims, num_bands, max_freq, concat_pos=False)
        output_no_concat = enc_no_concat(batch_size)
        expected_output_size_no_concat = batch_size, *index_dims, num_bands * len(index_dims) * 2
        self.assertEqual(output_no_concat.shape, expected_output_size_no_concat)

    def test_trainable_position_encoding(self):
        batch_size = 16
        index_dims = (8, 8)
        num_channels = 32

        # Test initialization
        enc = TrainablePositionEncoding(index_dims, num_channels)
        self.assertEqual(enc.index_dims, index_dims)
        self.assertEqual(enc.num_channels, num_channels)

        # Test forward pass
        output = enc(batch_size)
        expected_output_size = (batch_size, *index_dims, num_channels)
        self.assertEqual(output.shape, expected_output_size)

        # Test that weights are trainable
        initial_weights = enc.pos_embs.clone()
        optimizer = torch.optim.Adam(enc.parameters(), lr=0.01)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.allclose(initial_weights, enc.pos_embs))

    def test_position_encoding_projector(self):
        batch_size = 16
        index_dims = (8, 8)
        num_channels = 32
        projected_size = 64

        base_enc = TrainablePositionEncoding(index_dims, num_channels)
        projector = PositionEncodingProjector(projected_size, base_enc)

        # Test forward pass
        output = projector(batch_size)
        expected_output_size = (batch_size, *index_dims, projected_size)
        self.assertEqual(output.shape, expected_output_size)

    def test_build_position_encoding(self):
        index_dims = (8, 8)

        # Test Fourier encoding
        fourier_enc = build_position_encoding(
            "fourier", index_dims, fourier_position_encoding_kwargs={"num_bands": 4, "max_freq": 10.0}
        )
        self.assertIsInstance(fourier_enc, FourierPositionEncoding)

        # Test Trainable encoding
        trainable_enc = build_position_encoding("trainable", index_dims, trainable_position_encoding_kwargs={"num_channels": 32})
        self.assertIsInstance(trainable_enc, TrainablePositionEncoding)

        # Test projection
        projected_enc = build_position_encoding(
            "fourier", index_dims, project_pos_dim=64, fourier_position_encoding_kwargs={"num_bands": 4, "max_freq": 10.0}
        )
        self.assertIsInstance(projected_enc, PositionEncodingProjector)

        # Test invalid encoding type
        with self.assertRaises(ValueError):
            build_position_encoding("invalid_type", index_dims)


if __name__ == "__main__":
    unittest.main()
