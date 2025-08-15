"""Implements Adapter Controller, a module that keeps multiple
layers of Adapters, and controls which adapter layer to use."""
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation


class MetaLayersUpShareAdapterController(nn.Module):
    """Implements Meta Adapter controller module, in which
    the adapter layers' weights are generated from a unique hyper-network."""

    def __init__(self, config, input_dim):
        super().__init__()
        self.activation_type = config['non_linearity'].lower()
        if config['dropout'] > 0:
            self.dropout = nn.Dropout(p=config['dropout'])
        self.add_layer_norm_before_adapter = False
        self.add_layer_norm_after_adapter = False
        self.convert2fp16 = config['convert2fp16']

    def apply_layer_norm(self, inputs, layer_norm_weights):
        """Applies layer norm to the inputs."""
        normalized_shape = inputs.size(-1)
        if not self.convert2fp16:
            orig_type = inputs.dtype
            return torch.nn.functional.layer_norm(inputs.type(torch.float32), (normalized_shape,),
                                                  weight=layer_norm_weights.weight,
                                                  bias=layer_norm_weights.bias).type(orig_type)
        else:
            return torch.nn.functional.layer_norm(inputs, (normalized_shape,),
                                                  weight=layer_norm_weights.weight,
                                                  bias=layer_norm_weights.bias)

    def call_adapter(self, inputs, adapter_weights):
        """Computes the output of the adapter layers."""
        if self.convert2fp16:
            down = F.linear(inputs, weight=adapter_weights.down.weight,
                            bias=adapter_weights.down.bias)
            middle = get_activation(self.activation_type)(down)
            if hasattr(self, 'dropout'):
                middle = self.dropout(middle)

            if adapter_weights.up_share:
                if adapter_weights.up_unique:
                    output_unique = F.linear(middle, weight=adapter_weights.up_unique.weight,
                                             bias=adapter_weights.up_unique.bias)
                    output_share = F.linear(middle, weight=adapter_weights.up_share.weight,
                                            bias=adapter_weights.up_share.bias)
                    output = torch.cat([output_unique, output_share], dim=2)
                else:
                    output = F.linear(middle, weight=adapter_weights.up_share.weight,
                                      bias=adapter_weights.up_share.bias)
            else:
                output = F.linear(middle, weight=adapter_weights.up_unique.weight,
                                  bias=adapter_weights.up_unique.bias)
            return output
        else:
            orig_dtype = inputs.dtype
            down = F.linear(inputs.type(torch.float32), weight=adapter_weights.down.weight,
                            bias=adapter_weights.down.bias)
            middle = get_activation(self.activation_type)(down)
            if hasattr(self, 'dropout'):
                middle = self.dropout(middle)
            if adapter_weights.up_share:
                if adapter_weights.up_unique:
                    output_unique = F.linear(middle, weight=adapter_weights.up_unique.weight,
                                             bias=adapter_weights.up_unique.bias)
                    output_share = F.linear(middle, weight=adapter_weights.up_share.weight,
                                            bias=adapter_weights.up_share.bias)
                    output = torch.cat([output_unique, output_share], dim=2)
                else:
                    output = F.linear(middle, weight=adapter_weights.up_share.weight,
                                      bias=adapter_weights.up_share.bias)
            else:
                output = F.linear(middle, weight=adapter_weights.up_unique.weight,
                                  bias=adapter_weights.up_unique.bias)
            return output.type(orig_dtype)

    def forward(self, inputs, adapter_weights):

        z = self.apply_layer_norm(inputs, adapter_weights.pre_norm) if self.add_layer_norm_before_adapter else inputs
        outputs = self.call_adapter(z, adapter_weights)
        if self.add_layer_norm_after_adapter:
            outputs = self.apply_layer_norm(outputs, adapter_weights.post_norm)
        outputs = outputs + inputs
        return outputs

