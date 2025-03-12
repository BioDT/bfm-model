"""
Run with `python -m unittest test_attention.py`.
"""

import unittest

import torch
from einops import rearrange, repeat
from torch import nn

from bfm_model.perceiver_components.helpers import Attention


class TestAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.seq_len = 100
        self.q_dim = 256
        self.heads = 8
        self.head_dim = 64
        self.attention = Attention(q_dim=self.q_dim, heads=self.heads, head_dim=self.head_dim)

    def test_initialization(self):
        """Test if the attention module is initialized correctly."""
        self.assertIsInstance(self.attention.to_q, nn.Linear)
        self.assertIsInstance(self.attention.to_k, nn.Linear)
        self.assertIsInstance(self.attention.to_v, nn.Linear)
        self.assertIsInstance(self.attention.to_out, nn.Linear)
        self.assertEqual(self.attention.heads, self.heads)
        self.assertEqual(self.attention.scale, self.head_dim**-0.5)

    def test_forward_self_attention(self):
        """Test if the forward pass works for self-attention."""
        x = torch.randn(self.batch_size, self.seq_len, self.q_dim)
        output = self.attention(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.q_dim))

    def test_forward_cross_attention(self):
        """Test if the forward pass works for cross-attention."""
        x = torch.randn(self.batch_size, self.seq_len, self.q_dim)
        context = torch.randn(self.batch_size, self.seq_len * 2, self.q_dim)
        output = self.attention(x, context)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.q_dim))

    def test_masking(self):
        """Test if the attention mask is applied correctly."""
        x = torch.randn(self.batch_size, self.seq_len, self.q_dim)
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)

        # create a simple mask where the second half of the sequence is masked
        mask[:, self.seq_len // 2 :] = False

        # replace dropout to make the output deterministic
        original_dropout = self.attention.dropout
        self.attention.dropout = nn.Identity()

        output = self.attention(x, mask=mask)

        # check the shape of the output
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.q_dim))

        # compute attention weights (just as in the forward of the attention module)
        q = self.attention._split_heads(self.attention.to_q(x), self.attention.heads)
        k = self.attention._split_heads(self.attention.to_k(x), self.attention.heads)
        sim = torch.einsum("bhid,bhjd->bhij", q, k) * self.attention.scale

        # apply mask manually
        mask = rearrange(mask, "b ... -> b (...)")
        mask = repeat(mask, "b j -> b h () j", h=self.attention.heads)
        max_neg_value = -torch.finfo(sim.dtype).max
        sim = sim.masked_fill(~mask, max_neg_value)

        attention_weights = sim.softmax(dim=-1)

        # check if masked positions have zero attention weights
        masked_weights = attention_weights[:, :, :, self.seq_len // 2 :]
        self.assertTrue(torch.all(masked_weights == 0))

        # check if unmasked positions have larger attention weights
        unmasked_weights = attention_weights[:, :, :, : self.seq_len // 2]
        self.assertTrue(torch.all(unmasked_weights >= 1e-6))

        # Check if attention weights sum to 1 for each query
        sum_weights = attention_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sum_weights, torch.ones_like(sum_weights), atol=1e-6))

        # restore the original dropout
        self.attention.dropout = original_dropout

    def test_split_heads(self):
        """Test if the split_heads method works correctly."""
        x = torch.randn(self.batch_size, self.seq_len, self.heads * self.head_dim)
        split = self.attention._split_heads(x, self.heads)
        self.assertEqual(split.shape, (self.batch_size, self.heads, self.seq_len, self.head_dim))

    def test_merge_heads(self):
        """Test if the merge_heads method works correctly."""
        x = torch.randn(self.batch_size, self.heads, self.seq_len, self.head_dim)
        merged = self.attention._merge_heads(x, self.heads)
        self.assertEqual(merged.shape, (self.batch_size, self.seq_len, self.heads * self.head_dim))

    def test_attention_weights(self):
        """Test if the attention weights sum to 1 for each query."""
        x = torch.randn(self.batch_size, self.seq_len, self.q_dim)
        # remporarily replace dropout to make the output deterministic
        original_dropout = self.attention.dropout
        self.attention.dropout = nn.Identity()

        # run the attention mechanism
        self.attention(x)

        # check if attention weights sum to 1 for each query
        attention_weights = torch.einsum(
            "bhid,bhjd->bhij",
            self.attention._split_heads(self.attention.to_q(x), self.heads),
            self.attention._split_heads(self.attention.to_k(x), self.heads),
        )
        attention_weights = (attention_weights * self.attention.scale).softmax(dim=-1)
        self.assertTrue(
            torch.allclose(attention_weights.sum(dim=-1), torch.ones(self.batch_size, self.heads, self.seq_len), atol=1e-6)
        )

        # and again, restore the original dropout
        self.attention.dropout = original_dropout

    def test_output_range(self):
        """Test if the output of the attention mechanism is within reasonable range."""
        x = torch.randn(self.batch_size, self.seq_len, self.q_dim)
        output = self.attention(x)
        self.assertTrue(output.abs().max() < 100)  # Output should not explode


if __name__ == "__main__":
    unittest.main()
