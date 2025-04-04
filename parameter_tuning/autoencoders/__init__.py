"""
Autoencoders package for VAE model components.

This package contains components for building Variational Autoencoders (VAEs):
- Encoders: Convert input data to latent space distribution parameters
- Decoders: Convert latent vectors back to the original data space
- VAE: Combines encoders and decoders with the VAE training framework
"""

from .encoders import EncoderBase, MLPEncoder, RNNEncoder, LSTMEncoder
from .decoders import DecoderBase, MLPDecoder, RNNDecoder, LSTMDecoder
from .vae import VAE

__all__ = [
    'EncoderBase', 'MLPEncoder', 'RNNEncoder', 'LSTMEncoder',
    'DecoderBase', 'MLPDecoder', 'RNNDecoder', 'LSTMDecoder',
    'VAE'
]