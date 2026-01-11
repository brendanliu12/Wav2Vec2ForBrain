from __future__ import annotations

from abc import ABC, abstractmethod
from edit_distance import SequenceMatcher
from math import nan
from typing import Literal, Optional, cast
from types import SimpleNamespace
import inspect
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torcheval.metrics import WordErrorRate
from transformers import PreTrainedTokenizer, Wav2Vec2ProcessorWithLM

from sentence_transformers import SentenceTransformer

from src.datasets.batch_types import PhonemeSampleBatch, SampleBatch
from src.model.b2tmodel import ModelOutput
from src.train.history import DecodedPredictionBatch, MetricEntry, SingleEpochHistory
from src.util.phoneme_helper import PHONE_DEF_SIL

from src.model.brain_to_sentence_embedding import BrainToSentenceEmbeddingModel
from src.model.brain_feature_extractor import bfe_w_preprocessing_from_config


class Evaluator(ABC):
    def __init__(
        self,
        mode: Literal["train", "val", "test"],
        track_non_test_predictions: bool = False,
    ):
        self.running_loss = 0.0
        self.n_losses = 0
        self.latest_loss = nan
        self.mode = mode
        self.track_non_test_predictions = track_non_test_predictions

    def track_batch(self, predictions: ModelOutput, sample: SampleBatch):
        assert predictions.loss is not None
        self.running_loss += predictions.loss.item()
        self.n_losses += 1
        self.latest_loss = predictions.loss.item()
        self._track_batch(predictions, sample)

    def get_running_loss(self):
        return self.running_loss / self.n_losses

    def get_latest_loss(self):
        return self.latest_loss

    @abstractmethod
    def _track_batch(self, predictions: ModelOutput, sample: SampleBatch):
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self) -> SingleEpochHistory:
        raise NotImplementedError()

    def clean_up(self):
        pass


class DefaultEvaluator(Evaluator):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mode: Literal["train", "val", "test"],
        track_non_test_predictions: bool = False,
    ):
        super().__init__(mode, track_non_test_predictions)
        self.history = SingleEpochHistory()
        self.tokenizer = tokenizer

    def _track_batch(self, predictions: ModelOutput, sample: SampleBatch):
        predicted_strings, label_strings = self.decode_predictions(predictions, sample)

        predicted_strings = [self._cut_after_eos_token(s) for s in predicted_strings]

        additional_metrics: dict[str, float] = {}
        if label_strings is not None:
            additional_metrics["word_error_rate"] = (
                WordErrorRate().update(input=predicted_strings, target=label_strings).compute().item()
            )

            cer = self.calculate_char_error_rate(predicted_strings, label_strings)
            if cer == cer:  # not NaN
                additional_metrics["char_error_rate"] = cer

        predictions.metrics.update(additional_metrics)

        assert predictions.loss is not None, "Loss is None. Make sure to set loss in ModelOutput"
        self.history.add_batch_metric(
            MetricEntry(predictions.metrics, predictions.loss.cpu().item()),
            (
                DecodedPredictionBatch(predictions=predicted_strings, targets=label_strings)
                if self.mode == "test" or self.track_non_test_predictions
                else None
            ),
        )

    def evaluate(self) -> SingleEpochHistory:
        return self.history

    def decode_predictions(
        self, predictions: ModelOutput, sample: SampleBatch
    ) -> DecodedPredictionBatch:
        predicted_ids = predictions.logits.argmax(dim=-1).cpu().numpy()
        predicted_strings = self.tokenizer.batch_decode(predicted_ids, group_tokens=True)

        label_strings = (
            self.tokenizer.batch_decode(sample.target.cpu().numpy(), group_tokens=False)
            if sample.target is not None
            else None
        )
        return DecodedPredictionBatch(predicted_strings, label_strings)

    @staticmethod
    def _cut_after_eos_token(string: str) -> str:
        eos_token = "</s>"
        idx = string.find(eos_token)
        if idx != -1:
            return string[: (idx + len(eos_token))]
        return string

    @staticmethod
    def calculate_char_error_rate(predictions: list[str], targets: list[str]) -> float:
        total_seq_len = 0
        total_dist = 0
        for prediction, target in zip(predictions, targets):
            matcher = SequenceMatcher(a=target, b=prediction)
            dist = matcher.distance()
            if dist is not None:
                total_dist += dist
                total_seq_len += len(target)
        if total_seq_len > 0:
            return total_dist / total_seq_len
        return nan


class EnhancedDecodedBatch(DecodedPredictionBatch):
    predictions_lm_decoded: list[str]
    predictions_lm_decoded_topk: list[list[str]]


class EvaluatorWithW2vLMDecoder(DefaultEvaluator):
    """
    Extends DefaultEvaluator by optionally performing LM decoding (via Wav2Vec2ProcessorWithLM)
    at test time.

    If rerank_with_sentence_embeddings=True, will:
      1) request top-K hypotheses from LM decode (if supported by transformers version)
      2) run a separate BrainToSentenceEmbeddingModel on the SAME batch to get a brain-pred embedding
      3) embed candidate strings using SentenceTransformer
      4) choose the candidate with max cosine similarity to the brain embedding
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mode: Literal["train", "val", "test"],
        cache_dir: str,
        processor_checkpoint: str,
        track_non_test_predictions: bool = False,
        lm_decode_test_predictions: bool = False,
        lm_decode_beam_width: Optional[int] = None,
        lm_decode_beam_prune_logp: Optional[float] = None,
        lm_decode_token_min_logp: Optional[float] = None,
        lm_decode_alpha: Optional[float] = None,
        lm_decode_beta: Optional[float] = None,
        lm_decode_score_boundary: Optional[bool] = None,
        # NEW: reranking
        rerank_with_sentence_embeddings: bool = False,
        sentence_embedding_results_dir: Optional[str] = None,
        sentence_transformer_checkpoint: str = "sentence-transformers/all-mpnet-base-v2",
        sentence_rerank_top_k: int = 10,
    ):
        super().__init__(tokenizer, mode, track_non_test_predictions)

        # LM processor only needed in test mode, and only if enabled
        self.processor = (
            Wav2Vec2ProcessorWithLM.from_pretrained(processor_checkpoint, cache_dir=cache_dir)
            if lm_decode_test_predictions and mode == "test"
            else None
        )

        self.lm_decode_beam_width = lm_decode_beam_width
        self.lm_decode_beam_prune_logp = lm_decode_beam_prune_logp
        self.lm_decode_token_min_logp = lm_decode_token_min_logp
        self.lm_decode_alpha = lm_decode_alpha
        self.lm_decode_beta = lm_decode_beta
        self.lm_decode_score_boundary = lm_decode_score_boundary

        # rerank config
        self.rerank_with_sentence_embeddings = rerank_with_sentence_embeddings
        self.sentence_embedding_results_dir = sentence_embedding_results_dir
        self.sentence_transformer_checkpoint = sentence_transformer_checkpoint
        self.sentence_rerank_top_k = sentence_rerank_top_k

        # lazy-loaded models (only if reranking enabled)
        self._sent_encoder: Optional[SentenceTransformer] = None
        self._embed_model: Optional[BrainToSentenceEmbeddingModel] = None

    def _track_batch(self, predictions: ModelOutput, sample: SampleBatch):
        predicted_strings, label_strings = self.decode_predictions(predictions, sample)
        predicted_strings = [self._cut_after_eos_token(s) for s in predicted_strings]

        decoded_batch = EnhancedDecodedBatch(predictions=predicted_strings, targets=label_strings)
        decoded_batch.predictions_lm_decoded = []
        decoded_batch.predictions_lm_decoded_topk = []

        additional_metrics: dict[str, float] = {}

        # Base (non-LM) metrics
        if label_strings is not None:
            additional_metrics["word_error_rate"] = (
                WordErrorRate().update(input=predicted_strings, target=label_strings).compute().item()
            )
            cer = self.calculate_char_error_rate(predicted_strings, label_strings)
            if cer == cer:
                additional_metrics["char_error_rate"] = cer

        # LM decode only in test mode
        if self.processor is not None and self.mode == "test":
            logits_np = predictions.logits.detach().cpu().numpy()

            # Default: top-1 LM decode (original behavior)
            topk_texts: list[list[str]]

            sig = inspect.signature(self.processor.batch_decode)
            supports_n_best = "n_best" in sig.parameters

            if self.rerank_with_sentence_embeddings and supports_n_best:
                processed = self.processor.batch_decode(
                    logits_np,
                    beam_width=self.lm_decode_beam_width,
                    beam_prune_logp=self.lm_decode_beam_prune_logp,
                    token_min_logp=self.lm_decode_token_min_logp,
                    alpha=self.lm_decode_alpha,
                    beta=self.lm_decode_beta,
                    lm_score_boundary=self.lm_decode_score_boundary,
                    n_best=self.sentence_rerank_top_k,
                )
                # In this mode we expect List[List[str]]
                topk_texts = cast(list[list[str]], processed.text)
            else:
                processed = self.processor.batch_decode(
                    logits_np,
                    beam_width=self.lm_decode_beam_width,
                    beam_prune_logp=self.lm_decode_beam_prune_logp,
                    token_min_logp=self.lm_decode_token_min_logp,
                    alpha=self.lm_decode_alpha,
                    beta=self.lm_decode_beta,
                    lm_score_boundary=self.lm_decode_score_boundary,
                )
                # processed.text is List[str]
                topk_texts = [[t] for t in cast(list[str], processed.text)]

            decoded_batch.predictions_lm_decoded_topk = topk_texts

            # default chosen = LM best (top-1)
            chosen = [cands[0] if len(cands) > 0 else "" for cands in topk_texts]

            # Optional rerank (only if we actually have >1 candidate available)
            if self.rerank_with_sentence_embeddings and supports_n_best:
                device = predictions.logits.device
                self._init_rerank_models_if_needed(device=device)

                assert self._embed_model is not None
                assert self._sent_encoder is not None

                with torch.no_grad():
                    emb_out = self._embed_model.forward(sample)
                    brain_emb = F.normalize(emb_out.logits, dim=-1)  # [B, D]

                # embed all candidates
                flat = [s for group in topk_texts for s in group]
                cand_np = self._sent_encoder.encode(flat, normalize_embeddings=True)
                cand = torch.tensor(cand_np, device=device, dtype=brain_emb.dtype)  # [N, D]

                B = len(topk_texts)
                Kmax = max(len(g) for g in topk_texts) if B > 0 else 1
                D = cand.shape[-1]

                cand_emb = torch.zeros((B, Kmax, D), device=device, dtype=brain_emb.dtype)
                mask = torch.zeros((B, Kmax), device=device, dtype=torch.bool)

                idx = 0
                for i in range(B):
                    for j in range(len(topk_texts[i])):
                        cand_emb[i, j] = cand[idx]
                        mask[i, j] = True
                        idx += 1

                sims = (cand_emb * brain_emb.unsqueeze(1)).sum(-1)  # [B, Kmax]
                sims = sims.masked_fill(~mask, float("-inf"))
                best_j = torch.argmax(sims, dim=1).tolist()
                chosen = [
                    topk_texts[i][best_j[i]] if len(topk_texts[i]) > 0 else "" for i in range(B)
                ]

            decoded_batch.predictions_lm_decoded = chosen

            # LM-decode metrics (only if targets exist)
            if label_strings is not None:
                additional_metrics["word_error_rate_lm_decode"] = (
                    WordErrorRate().update(input=chosen, target=label_strings).compute().item()
                )
                additional_metrics["char_error_rate_lm_decode"] = self.calculate_char_error_rate(
                    chosen, label_strings
                )

        # update metrics + history
        predictions.metrics.update(additional_metrics)

        assert predictions.loss is not None, "Loss is None. Make sure to set loss in ModelOutput"
        self.history.add_batch_metric(
            MetricEntry(predictions.metrics, predictions.loss.cpu().item()),
            decoded_batch if (self.mode == "test" or self.track_non_test_predictions) else None,
        )

    def _init_rerank_models_if_needed(self, device: torch.device):
        if self._sent_encoder is None:
            self._sent_encoder = SentenceTransformer(self.sentence_transformer_checkpoint)

        if self._embed_model is None:
            assert self.sentence_embedding_results_dir is not None, (
                "rerank_with_sentence_embeddings=True but sentence_embedding_results_dir=None"
            )

            cfg_path = os.path.join(self.sentence_embedding_results_dir, "config.json")
            model_path = os.path.join(self.sentence_embedding_results_dir, "model.pt")

            assert os.path.exists(cfg_path), f"Missing {cfg_path}"
            assert os.path.exists(model_path), f"Missing {model_path}"

            with open(cfg_path, "r") as f:
                cfg_dict = json.load(f)

            cfg = SimpleNamespace(**cfg_dict)

            brain_encoder = bfe_w_preprocessing_from_config(
                cfg,
                getattr(cfg, "brain_encoder_path", None),
                getattr(cfg, "wav2vec_checkpoint"),
            )

            emb_dim = 768  # all-mpnet-base-v2
            self._embed_model = BrainToSentenceEmbeddingModel(
                brain_encoder=brain_encoder,
                emb_dim=emb_dim,
                use_cosine_loss=True,
            )

            state = torch.load(model_path, map_location="cpu")
            self._embed_model.load_state_dict(state, strict=True)
            self._embed_model.eval()
            self._embed_model.to(device)


class B2PEvaluator(Evaluator):
    def __init__(
        self,
        mode: Literal["train", "val", "test"],
        track_non_test_predictions: bool = False,
    ):
        super().__init__(mode, track_non_test_predictions)
        self.history = SingleEpochHistory()

    def _track_batch(self, predictions: ModelOutput, sample: PhonemeSampleBatch):
        phoneme_error_rate, prediction_batch = self._calc_phoneme_error_rate(sample, predictions)
        additional_metrics = {"phoneme_error_rate": phoneme_error_rate}
        predictions.metrics.update(additional_metrics)

        assert predictions.loss is not None, "Loss is None. Make sure to set loss in ModelOutput"
        self.history.add_batch_metric(
            MetricEntry(predictions.metrics, predictions.loss.cpu().item()),
            prediction_batch if (self.mode == "test" or self.track_non_test_predictions) else None,
        )

    def evaluate(self) -> SingleEpochHistory:
        return self.history

    def _calc_phoneme_error_rate(self, batch: PhonemeSampleBatch, predictions: ModelOutput):
        pred = predictions.logits
        total_edit_distance = 0
        total_seq_length = 0
        labels = []
        predicted = []

        for iterIdx in range(pred.shape[0]):
            if batch.target is None:
                continue

            decodedSeq = torch.argmax(torch.tensor(pred[iterIdx, :, :]), dim=-1)
            decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
            decodedSeq = decodedSeq.cpu().detach().numpy()
            decodedSeq = np.array([i for i in decodedSeq if i != 0])

            trueSeq = np.array([idx.item() for idx in batch.target[iterIdx].cpu() if idx >= 0])

            labels.append([PHONE_DEF_SIL[i - 1] for i in trueSeq])
            predicted.append([PHONE_DEF_SIL[i - 1] for i in decodedSeq])

            matcher = SequenceMatcher(a=trueSeq.tolist(), b=decodedSeq.tolist())
            dist = matcher.distance()
            if dist is None:
                print("[evaluate batch]: distance from sequence matcher is None, skipping.")
                continue
            total_edit_distance += dist
            total_seq_length += len(trueSeq)

        return (
            total_edit_distance / total_seq_length if total_seq_length > 0 else nan,
            DecodedPredictionBatch(predicted, labels),
        )
