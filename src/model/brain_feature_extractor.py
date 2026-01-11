from typing import Optional

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pydantic import BaseModel, Field

from src.args.base_args import PRETRAINED_LATENT_SIZES
from src.datasets.batch_types import PhonemeSampleBatch, SampleBatch
from src.model.b2p2t_model import B2P2TModel, B2P2TModelArgsModel
from src.model.b2tmodel import B2TModel, ModelOutput
from src.util.nn_helper import ACTIVATION_FUNCTION, create_fully_connected


class BrainFeatureExtractorArgsModel(BaseModel):
    encoder_rnn_hidden_size: int = Field(default=256, alias="encoder_gru_hidden_size")
    encoder_bidirectional: bool = True
    encoder_num_rnn_layers: int = Field(default=2, alias="encoder_num_gru_layers")
    encoder_bias: bool = True
    encoder_dropout: float = 0.0
    # encoder_learnable_inital_state: bool = False
    encoder_learnable_initial_hidden_state: bool = Field(
        default=False, alias="encoder_learnable_inital_state"
    )
    encoder_learnable_initial_cell_state: bool = False  # Only for LSTM
    encoder_random_initial_hidden_state: bool = Field(
        default=False, alias="encoder_random_inital_state"
    )
    encoder_random_initial_cell_state: bool = False  # Only for LSTM
    encoder_fc_hidden_sizes: list[int] = []
    encoder_fc_activation_function: ACTIVATION_FUNCTION = "gelu"
    encoder_rnn_type: str = "gru"  # gru or lstm


class BrainFeatureExtractor(torch.nn.Module):
    def __init__(
        self, config: BrainFeatureExtractorArgsModel, in_size, wav2vec_checkpoint: str
    ):
        super().__init__()
        self.config = config
        self.num_directions = 2 if config.encoder_bidirectional else 1

        if config.encoder_rnn_type not in ["gru", "lstm"]:
            raise ValueError(
                f"Invalid encoder_rnn_type: {config.encoder_rnn_type}. Must be 'gru' or 'lstm'."
            )
        if config.encoder_rnn_type == "lstm":
            if config.encoder_random_initial_cell_state:
                self.c_start = torch.nn.Parameter(
                    torch.randn(
                        self.num_directions * config.encoder_num_rnn_layers,
                        config.encoder_rnn_hidden_size,
                    ),
                    requires_grad=config.encoder_learnable_initial_cell_state,
                )
            else:
                self.c_start = torch.nn.Parameter(
                    torch.zeros(
                        self.num_directions * config.encoder_num_rnn_layers,
                        config.encoder_rnn_hidden_size,
                    ),
                    requires_grad=config.encoder_learnable_initial_cell_state,
                )
        if config.encoder_random_initial_hidden_state:
            self.h_start = torch.nn.Parameter(
                torch.randn(
                    self.num_directions * config.encoder_num_rnn_layers,
                    config.encoder_rnn_hidden_size,
                ),
                requires_grad=config.encoder_learnable_initial_hidden_state,
            )
        else:
            self.h_start = torch.nn.Parameter(
                torch.zeros(
                    self.num_directions * config.encoder_num_rnn_layers,
                    config.encoder_rnn_hidden_size,
                ),
                requires_grad=config.encoder_learnable_initial_hidden_state,
            )

        self.rnn: torch.nn.LSTM | torch.nn.GRU
        if config.encoder_rnn_type == "lstm":
            self.rnn = torch.nn.LSTM(
                in_size,
                config.encoder_rnn_hidden_size,
                config.encoder_num_rnn_layers,
                dropout=config.encoder_dropout,
                bias=config.encoder_bias,
                bidirectional=config.encoder_bidirectional,
                batch_first=True,
            )
        else:
            self.rnn = torch.nn.GRU(
                in_size,
                config.encoder_rnn_hidden_size,
                config.encoder_num_rnn_layers,
                dropout=config.encoder_dropout,
                bias=config.encoder_bias,
                bidirectional=config.encoder_bidirectional,
                batch_first=True,
            )

        self.fc = create_fully_connected(
            config.encoder_rnn_hidden_size * self.num_directions,
            PRETRAINED_LATENT_SIZES[wav2vec_checkpoint],
            config.encoder_fc_hidden_sizes,
            config.encoder_fc_activation_function,
        )

    def forward(self, batch: PhonemeSampleBatch) -> torch.Tensor:
        x = batch.input
        batch_size = x.shape[0]

        # If we have lengths, pack so the RNN ignores padded timesteps entirely.
        lengths = getattr(batch, "input_lens", None)
        if lengths is not None:
            # pack_padded_sequence expects CPU lengths
            lengths_cpu = lengths.to("cpu")
            packed_x = pack_padded_sequence(
                x,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False,
            )

            if self.config.encoder_rnn_type == "lstm":
                packed_out, _ = self.rnn(
                    packed_x,
                    (
                        self.h_start.unsqueeze(1).repeat(1, batch_size, 1),
                        self.c_start.unsqueeze(1).repeat(1, batch_size, 1),
                    ),
                )
            else:
                packed_out, _ = self.rnn(
                    packed_x,
                    self.h_start.unsqueeze(1).repeat(1, batch_size, 1),
                )

            # Unpack back to padded tensor [B, T_max, hidden*dir]
            out, _ = pad_packed_sequence(
                packed_out,
                batch_first=True,
            )
        else:
            # Fallback: no lengths provided, run normally (uses padding as real input)
            if self.config.encoder_rnn_type == "lstm":
                out, _ = self.rnn(
                    x,
                    (
                        self.h_start.unsqueeze(1).repeat(1, batch_size, 1),
                        self.c_start.unsqueeze(1).repeat(1, batch_size, 1),
                    ),
                )
            else:
                out, _ = self.rnn(
                    x,
                    self.h_start.unsqueeze(1).repeat(1, batch_size, 1),
                )

        out = self.fc(out)
        return out


class B2TBrainFeatureExtractor(B2TModel):
    def __init__(
        self,
        config: BrainFeatureExtractorArgsModel,
        wav2vec_checkpoint: str,
        in_size: int,
    ):
        super().__init__()
        self.encoder = BrainFeatureExtractor(
            config,
            in_size,
            wav2vec_checkpoint,
        )

    def forward(self, batch: SampleBatch) -> ModelOutput:
        out = self.encoder(batch)
        return ModelOutput(logits=out, metrics={})


class B2P2TBrainFeatureExtractorArgsModel(
    BrainFeatureExtractorArgsModel, B2P2TModelArgsModel
):
    pass


def bfe_w_preprocessing_from_config(
    config: B2P2TBrainFeatureExtractorArgsModel,
    brain_encoder_path: Optional[str],
    wav2vec_checkpoint: str,
):
    brain_feat_extractor = B2P2TModel(
        config,
        B2TBrainFeatureExtractor(
            config,
            wav2vec_checkpoint,
            B2P2TModel.get_in_size_after_preprocessing(config.unfolder_kernel_len),
        ),
    ).cuda()
    if brain_encoder_path != None:
        state = torch.load(brain_encoder_path, map_location="cuda")
        unneeded_keys = [
            key
            for key in state.keys()
            if key.startswith("neural_decoder.discriminator")
            or key.startswith("neural_decoder.suc_for_ctc")
        ]
        for key in unneeded_keys:
            del state[key]
        brain_feat_extractor.load_state_dict(
            state,
            strict=True,
        )
    return brain_feat_extractor
