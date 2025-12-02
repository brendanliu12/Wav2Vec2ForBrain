from pydantic import BaseModel
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerPreTrainedModel,
    Wav2Vec2ConformerConfig,
    Wav2Vec2ConformerEncoder,
    Wav2Vec2ConformerAdapter,
    Wav2Vec2ConformerForCTC,
    Wav2Vec2BaseModelOutput,
)
import torch
from typing import Optional, cast
from src.datasets.batch_types import B2tSampleBatch
from src.model.b2tmodel import B2TModel, ModelOutput


class W2VConformerBrainEncoderModel(B2TModel):
    def __init__(
        self,
        brain_encoder: B2TModel,
        wav2vec_checkpoint: str,
        use_area44_sentence_bias: bool = False,
        area44_input_dim: int | None = None,
        area44_hidden_dim: int = 256,
        vocab_size: int | None = None,
    ):
        super().__init__()
        self.brain_encoder = brain_encoder
        self.use_area44_sentence_bias = use_area44_sentence_bias

        w2v_config = cast(
            Wav2Vec2ConformerConfig,
            Wav2Vec2ConformerConfig.from_pretrained(wav2vec_checkpoint),
        )
        self.w2v_encoder = cast(
            Wav2Vec2ConformerWithoutFeatExtrForCTC,
            Wav2Vec2ConformerWithoutFeatExtrForCTC.from_pretrained(
                wav2vec_checkpoint, config=w2v_config
            ),
        )
        self.loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

        # Optional sentence-level encoder for area 44
        if self.use_area44_sentence_bias:
            assert (
                area44_input_dim is not None and vocab_size is not None
            ), "area44_input_dim and vocab_size must be provided when use_area44_sentence_bias=True"

            # Simple bidirectional GRU over area-44 sequence
            self.area44_rnn = torch.nn.GRU(
                input_size=area44_input_dim,
                hidden_size=area44_hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            # Map sentence embedding to vocab-level bias
            self.area44_to_vocab = torch.nn.Linear(
                2 * area44_hidden_dim, vocab_size
            )
            self.area44_dropout = torch.nn.Dropout(p=0.1)


    def forward(self, batch: B2tSampleBatch):
        encoded_brain = self.brain_encoder.forward(batch)
        targets = batch.target
        assert targets is not None

        # mask out padding tokens for CTC
        targets = torch.where(targets < 1, torch.tensor(-100, device=targets.device), targets)

        # base CTC logits from wav2vec2-conformer
        w2v_output = self.w2v_encoder.forward(encoded_brain.logits)  # [B, T, V]

        # --- NEW: add sentence-level bias from area 44 ---
        if self.use_area44_sentence_bias:
            assert hasattr(batch, "input_44"), "Batch missing input_44 but use_area44_sentence_bias=True"

            x44 = batch.input_44  # [B, T44, F44]
            # Optional: respect true lengths via packing
            if hasattr(batch, "input_44_lens"):
                lengths = batch.input_44_lens
                # pack_padded_sequence expects CPU lengths
                packed = torch.nn.utils.rnn.pack_padded_sequence(
                    x44,
                    lengths.cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
                _, h_n = self.area44_rnn(packed)
            else:
                # fall back: unmasked
                _, h_n = self.area44_rnn(x44)

            # h_n shape: [num_layers * num_directions, B, hidden_dim]
            # num_layers=1, bidirectional=True -> shape [2, B, H]
            # concatenate last forward & backward states
            h_forward = h_n[-2]  # [B, H]
            h_backward = h_n[-1]  # [B, H]
            h_sentence = torch.cat([h_forward, h_backward], dim=-1)  # [B, 2H]

            bias_44 = self.area44_to_vocab(
                self.area44_dropout(h_sentence)
            )  # [B, V]
            # broadcast bias over time dimension
            bias_44 = bias_44.unsqueeze(1).expand(
                -1, w2v_output.size(1), -1
            )  # [B, T, V]

            w2v_output = w2v_output + bias_44
        # --- end NEW ---

        ctc_loss = (
            self.loss.forward(
                torch.log_softmax(w2v_output, -1).transpose(0, 1),
                targets,
                encoded_brain.logit_lens.cuda(),
                batch.target_lens.cuda(),
            )
            if batch.target_lens is not None and encoded_brain.logit_lens is not None
            else None
        )

        return ModelOutput(
            w2v_output,
            {"ctc_loss": ctc_loss.item()} if ctc_loss is not None else {},
            loss=ctc_loss,
        )



class Wav2Vec2ConformerWithoutFeatExtrForCTC(Wav2Vec2ConformerForCTC):
    def __init__(self, config, target_lang: Optional[str] = None):
        super().__init__(config, target_lang)
        self.wav2vec2_conformer = Wav2Vec2ConformerWithoutFeatExtrModel(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.wav2vec2_conformer(
            x,
            return_dict=True,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)
        return logits


class Wav2Vec2ConformerWithoutFeatExtrModel(Wav2Vec2ConformerPreTrainedModel):
    def __init__(self, config: Wav2Vec2ConformerConfig):
        super().__init__(config)
        self.config = config
        self.encoder = Wav2Vec2ConformerEncoder(config)
        self.adapter = Wav2Vec2ConformerAdapter(config) if config.add_adapter else None

        self.post_init()

    def forward(
        self,
        input_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Wav2Vec2BaseModelOutput:
        encoder_outputs = self.encoder(
            input_values,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = encoder_outputs[0]
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=input_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
