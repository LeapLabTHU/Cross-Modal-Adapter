"""Defines the output class for the adapter layers' parameters."""
import torch
from dataclasses import dataclass


@dataclass
class SamplerOutput:
    """Base class for the base and weights of each adapter."""
    weight: torch.FloatTensor = None
    bias: torch.FloatTensor = None


@dataclass
class LayerNormOutput:
    """Base class for the base and weights of the conditional
    layer norms."""
    weight: torch.FloatTensor = None
    bias: torch.FloatTensor = None


@dataclass
class AdapterOutput:
    """Base class for each adapter weights"""
    up: SamplerOutput = None
    down: SamplerOutput = None
    pre_norm: LayerNormOutput = None
    post_norm: LayerNormOutput = None


@dataclass
class AdapterBlockOutput:
    """
    Base class for adapter layer's outputs.
    """
    feed_forward: AdapterOutput = None
    self_attention: AdapterOutput = None


@dataclass
class UpSharedAdapterOutput:
    """Base class for each adapter weights"""
    up_unique: SamplerOutput = None
    up_share: SamplerOutput = None
    down: SamplerOutput = None
    pre_norm: LayerNormOutput = None
    post_norm: LayerNormOutput = None


@dataclass
class UpSharedAdapterBlockOutput:
    """
    Base class for adapter layer's outputs.
    """
    self_attention: UpSharedAdapterOutput = None
    feed_forward: UpSharedAdapterOutput = None


@dataclass
class DownSharedAdapterOutput:
    """Base class for each adapter weights"""
    down_unique: SamplerOutput = None
    down_share: SamplerOutput = None
    up: SamplerOutput = None
    pre_norm: LayerNormOutput = None
    post_norm: LayerNormOutput = None


@dataclass
class DownSharedAdapterBlockOutput:
    """
    Base class for adapter layer's outputs.
    """
    self_attention: DownSharedAdapterOutput = None
    feed_forward: DownSharedAdapterOutput = None


@dataclass
class STAdapterOutput:
    """Base class for each adapter weights"""
    up: SamplerOutput = None
    down: SamplerOutput = None
    dwconv: SamplerOutput = None
    pre_norm: LayerNormOutput = None
    post_norm: LayerNormOutput = None


@dataclass
class DownUpSharedAdapterOutput:
    """Base class for each adapter weights"""
    down_unique: SamplerOutput = None
    down_share: SamplerOutput = None
    up_unique: SamplerOutput = None
    up_share: SamplerOutput = None
    pre_norm: LayerNormOutput = None
    post_norm: LayerNormOutput = None


@dataclass
class DownUpSharedAdapterBlockOutput:
    """
    Base class for adapter layer's outputs.
    """
    self_attention: DownUpSharedAdapterOutput = None
    feed_forward: DownUpSharedAdapterOutput = None