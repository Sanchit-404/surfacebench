import torch
import torch.nn as nn


class SymbolicEquation(nn.Module):

    def __init__(self, {param_inputs}):
        """
        Define trainable continuous parameters for differentiable optimization.
        Tentatively initialize the parameters with the default values in args.

        Args:
            {param_desc}
        """
        super().__init__()
        {param_init}

    def forward(self, {input_variables}) -> torch.Tensor:
        {forward_function_description}