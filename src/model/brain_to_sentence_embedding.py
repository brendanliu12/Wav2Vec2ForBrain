import torch
from torch import nn
import torch.nn.functional as F

from src.model.b2tmodel import B2TModel, ModelOutput
from src.datasets.batch_types import B2tSampleBatch
from src.model.b2p2t_model import B2P2TModel
from src.args.base_args import PRETRAINED_LATENT_SIZES


class BrainToSentenceEmbeddingModel(B2TModel):
    def __init__(
        self,
        brain_encoder: B2P2TModel,
        emb_dim: int,
        use_cosine_loss: bool = True,
    ):
        super().__init__()
        self.brain_encoder = brain_encoder
        self.emb_dim = emb_dim
        self.use_cosine_loss = use_cosine_loss

        # 🔹 The B2P2T brain encoder outputs a latent dim that depends on wav2vec checkpoint,
        #    not directly on encoder_rnn_hidden_size.
        wav2v_ckpt = brain_encoder.config.wav2vec_checkpoint
        if wav2v_ckpt not in PRETRAINED_LATENT_SIZES:
            raise ValueError(
                f"Unknown wav2vec checkpoint '{wav2v_ckpt}' for latent size; "
                f"add it to PRETRAINED_LATENT_SIZES in base_args.py."
            )
        latent_dim = PRETRAINED_LATENT_SIZES[wav2v_ckpt]  # e.g., 768 or 1024

        # 🔹 Pool over time then project into the sentence-embedding space
        # self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, emb_dim),
        )
        
    def forward(self, batch: B2tSampleBatch) -> ModelOutput:
        # 1) Encode brain activity with the B2P2T brain encoder
        encoded = self.brain_encoder.forward(batch)
        H = encoded.logits  # [B, T', D] where D = latent_dim

        # 2) Length-aware mean pooling (ignore padded timesteps)
        # Prefer encoder-provided lengths (B2P2TModel sets out.logit_lens),
        # otherwise fall back to batch.input_lens.
        lens = getattr(encoded, "logit_lens", None)
        if lens is None:
            lens = getattr(batch, "input_lens", None)

        if lens is None:
            # Fallback: no lengths available, average over all timesteps
            pooled = H.mean(dim=1)  # [B, D]
        else:
            # H: [B, T', D]
            B, Tp, D = H.shape
            lens = lens.to(H.device).clamp(min=1)

            # mask: [B, T'] where True = real timestep
            t = torch.arange(Tp, device=H.device).unsqueeze(0)  # [1, T']
            mask = (t < lens.unsqueeze(1)).unsqueeze(-1)        # [B, T', 1]

            pooled = (H * mask).sum(dim=1) / mask.sum(dim=1)    # [B, D]


        # 3) Project to semantic embedding space
        pred_emb = self.proj(pooled)          # [B, emb_dim]

        # 4) Supervise with precomputed sentence embeddings (if available)
        target_emb = batch.target_embedding  # can be None at inference time

        if target_emb is None:
            # Inference mode (used for reranking): no loss, no metrics
            return ModelOutput(
                logits=pred_emb,
                metrics={},
                loss=None,
            )

        # Training mode
        target_emb = target_emb.to(pred_emb.device)

        if self.use_cosine_loss:
            pred_norm = F.normalize(pred_emb, dim=-1)
            target_norm = F.normalize(target_emb, dim=-1)
            cos_sim = (pred_norm * target_norm).sum(dim=-1)  # [B]
            loss = 1.0 - cos_sim.mean()
            metrics = {"cosine_similarity": cos_sim.mean().item()}
        else:
            loss = F.mse_loss(pred_emb, target_emb)
            metrics = {"mse": loss.item()}

        return ModelOutput(
            logits=pred_emb,
            metrics=metrics,
            loss=loss,
        )


