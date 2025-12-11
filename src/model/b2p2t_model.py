from pydantic import BaseModel
from src.datasets.batch_types import B2tSampleBatch

from src.model.b2tmodel import B2TModel, ModelOutput
from typing import Literal
import torch
from torch import nn
from pydantic import BaseModel
from src.model.b2tmodel import B2TModel, ModelOutput
from typing import Literal, cast
import torch
from torch import nn
import math
from torch.nn import functional as F


DEFAULT_UNFOLDER_KERNEL_LEN = 32


class B2P2TModelArgsModel(BaseModel):
    input_layer_nonlinearity: Literal["softsign"] = "softsign"
    unfolder_kernel_len: int = DEFAULT_UNFOLDER_KERNEL_LEN
    unfolder_stride_len: int = 4
    gaussian_smooth_width: float = 0.3


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size: int, sigma: float, dim=2):
        super(GaussianSmoothing, self).__init__()

        kernel_sizes = [kernel_size] * dim
        sigmas = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_sizes]
        )
        for size, std, mgrid in zip(kernel_sizes, sigmas, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(cast(torch.Tensor, kernel))

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding="same")


class B2P2TModel(B2TModel):
    """Wraps the neural decoder model to perform trial day dependant preprocessing"""

    def __init__(
        self, config: B2P2TModelArgsModel, neural_decoder: B2TModel, pad_token_id=0
    ):
        super().__init__()
        self.config = config
        self.pad_token_id = pad_token_id

        if config.input_layer_nonlinearity != "softsign":
            raise NotImplementedError(
                "Only softsign is currently supported as input layer nonlinearity"
            )
        self.input_layer_nonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (config.unfolder_kernel_len, 1),
            dilation=1,
            padding=0,
            stride=config.unfolder_stride_len,
        )
        n_days = 24
        neural_dim_len = 256
        self.gaussian_smoother = GaussianSmoothing(
            neural_dim_len, 20, config.gaussian_smooth_width, dim=1
        )
        self.day_weights = torch.nn.Parameter(
            torch.randn(n_days, neural_dim_len, neural_dim_len)
        )
        self.day_bias = torch.nn.Parameter(torch.zeros(n_days, 1, neural_dim_len))
        for x in range(n_days):
            self.day_weights.data[x, :, :] = torch.eye(neural_dim_len)

        self.neural_decoder = neural_decoder

        # Input layers
        for x in range(n_days):
            setattr(
                self, "inpLayer" + str(x), nn.Linear(neural_dim_len, neural_dim_len)
            )

        for x in range(n_days):
            layer = getattr(self, "inpLayer" + str(x))
            layer.weight = torch.nn.Parameter(layer.weight + torch.eye(neural_dim_len))

    def forward(self, batch: B2tSampleBatch) -> ModelOutput:
        """
        batch can also be of sub type PhonemeSampleBatch
        """
        x = batch.input
        targets = getattr(batch, "target", None)
        day_idxs = batch.day_idxs

        if targets is not None:
            targets = torch.where(
                targets == self.pad_token_id, torch.tensor(-100), targets
            )

        neural_input = torch.permute(x, (0, 2, 1))
        neural_input = self.gaussian_smoother(neural_input)
        neural_input = torch.permute(neural_input, (0, 2, 1))

        # apply day layer
        day_weights = torch.index_select(self.day_weights, 0, day_idxs)
        transformed_neural = torch.einsum(
            "btd,bdk->btk", neural_input, day_weights
        ) + torch.index_select(self.day_bias, 0, day_idxs)
        transformed_neural = self.input_layer_nonlinearity(transformed_neural)

        # stride/kernel
        strided_inputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformed_neural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )
        preprocessed_batch = batch.copy_and_change(input=strided_inputs)
        if hasattr(batch, "input_lens"):
            processed_in_lens = (
                (batch.input_lens - self.config.unfolder_kernel_len)
                / self.config.unfolder_stride_len
            ).to(torch.int32)
            preprocessed_batch.input_lens = processed_in_lens
            out = self.neural_decoder.forward(preprocessed_batch)
            out.logit_lens = processed_in_lens
            return out

        return self.neural_decoder.forward(preprocessed_batch)

    @classmethod
    def get_in_size_after_preprocessing(cls, unfolder_kernel_len: int):
        return (256) * unfolder_kernel_len
