import torch
import torch.nn as nn


class SymbolicEquation(nn.Module):

    def __init__(self, {param_variables}):
        """
        Define trainable continuous physical parameters for differentiable optimization.
        Tentatively initialize the parameters with the default values in args.

        Args:
            {param_variables_desc}
        """

        super().__init__()
        {init_params}

    def forward(self, {input_variables}) -> torch.Tensor:
        {forward_function_description}