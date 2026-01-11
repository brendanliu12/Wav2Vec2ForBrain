from typing import Any, Literal, cast

from git import Optional
from pyctcdecode.constants import (
    DEFAULT_BEAM_WIDTH,
    DEFAULT_MIN_TOKEN_LOGP,
    DEFAULT_PRUNE_LOGP,
)
from pydantic import Field
import torch
from src.model.brain_feature_extractor import B2P2TBrainFeatureExtractorArgsModel
from src.model.w2v_conformer_custom_feat_extractor import W2VConformerBrainEncoderModel
from src.experiments.b2t_experiment import B2TArgsModel, B2TExperiment
from src.model.brain_feature_extractor import (
    bfe_w_preprocessing_from_config,
)
from src.model.w2v_custom_feat_extractor import (
    W2VBrainEncoderModel,
)
from src.args.yaml_config import YamlConfigModel
from torch.optim.optimizer import Optimizer
from src.train.evaluator import EvaluatorWithW2vLMDecoder
from src.util.warmup_scheduler import get_2module_warmup_scheduler

# Baseline Experiment: b2p2t_gru:
# 5gram + rescoring: 0.28 WER (/hpi/fs00/scratch/tobias.fiedler/brain2text/experiment_results/b2p2t_gru/2024-03-27_17#31#07),
# 3gram: 0.3153 WER
# (/hpi/fs00/scratch/tobias.fiedler/brain2text/experiment_results/b2p2t_gru/2024-03-28_08#18#39)


W2V_CHECKPOINT_TO_PROCESSOR = {
    "facebook/wav2vec2-conformer-rope-large-960h-ft": "patrickvonplaten/wav2vec2-base-100h-with-lm",  # TODO: can we use this here?
}


class B2TGruAndW2VConformerArgsModel(B2TArgsModel, B2P2TBrainFeatureExtractorArgsModel):
    brain_encoder_path: Optional[str] = None
    unfreeze_strategy: Literal["brain_encoder", "brain_encoder+w2v"] = "brain_encoder"
    w2v_learning_rate: Optional[float] = None
    w2v_warmup_start_step: Optional[int] = Field(
        default=None,
        description="Epoch at which warm up phase of w2v lr starts. Before LR will be 0. 0 if not provided",
    )
    w2v_warmup_steps: Optional[int] = Field(
        default=None,
        description="Num epochs from w2v_warmup_start_step to reach full w2v_learning_rate. 0 if not provided",
    )
    wav2vec_checkpoint: str = "facebook/wav2vec2-conformer-rope-large-960h-ft"
    lm_decode_test_predictions: bool = False
    adjust_global_lr_to_w2v_postwarmup_lr: Optional[bool] = Field(
        description="Adjust the global learning rate to that of w2v over w2v warmup interval, then keep at w2v_learning_rate. Only valid when brain_encoder+w2v unfreeze strategy is set."
    )
    lm_decode_beam_width: int = DEFAULT_BEAM_WIDTH
    lm_decode_beam_prune_logp: float = DEFAULT_PRUNE_LOGP
    lm_decode_token_min_logp: float = DEFAULT_MIN_TOKEN_LOGP
    lm_decode_alpha: float = 0.5
    lm_decode_beta: float = 0.5
    lm_score_boundary: bool = False
    # --- Rerank LM N-best using a separate sentence-embedding model ---
    rerank_with_sentence_embeddings: bool = False

    # Path to the *results dir* produced by b2p2t_44_sentence_embedding
    # (must contain model.pt and config.json)
    sentence_embedding_results_dir: Optional[str] = None

    # SentenceTransformer used to embed candidate hypotheses
    sentence_transformer_checkpoint: str = "sentence-transformers/all-mpnet-base-v2"

    # Number of candidates to take from LM decode
    sentence_rerank_top_k: int = 10


class B2TGruAndW2VConformerExperiment(B2TExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)

        if self.config.tokenizer_checkpoint != self.config.wav2vec_checkpoint:
            print(
                f"Tokenizer checkpoint ({self.config.tokenizer_checkpoint}) is different to wav2vec_checkpoint ({self.config.wav2vec_checkpoint}). This may lead to unexpected behaviour"
            )

    def get_name(self) -> str:
        return "b2p2t_gru+w2v_conformer"

    @staticmethod
    def get_args_model():
        return B2TGruAndW2VConformerArgsModel

    def _create_model(self):
        brain_encoder = bfe_w_preprocessing_from_config(
            self.config, self.config.brain_encoder_path, self.config.wav2vec_checkpoint
        )
        model = W2VConformerBrainEncoderModel(
            brain_encoder, self.config.wav2vec_checkpoint
        )
        return model

    def create_optimizer(self) -> Optimizer:
        def get_trainable_params():
            if self.config.unfreeze_strategy == "brain_encoder+w2v":
                return [
                    {
                        "params": cast(
                            W2VBrainEncoderModel, self.model
                        ).brain_encoder.parameters()
                    },
                    {
                        "params": cast(
                            W2VBrainEncoderModel, self.model
                        ).w2v_encoder.parameters(),
                        "lr": (
                            self.config.w2v_learning_rate
                            if self.config.w2v_learning_rate is not None
                            else self.config.learning_rate
                        ),
                    },
                ]
            if self.config.unfreeze_strategy == "brain_encoder":
                assert (
                    self.config.w2v_learning_rate is None
                ), "w2v_learning_rate can only be set if unfreeze strategy is brain_encoder+w2v"
                return cast(W2VBrainEncoderModel, self.model).brain_encoder.parameters()
            raise Exception(
                f"Unfreeze strategy {self.config.unfreeze_strategy} is not implemented for wav2vec experiment"
            )

        optim_cls: Any = self._get_optimizer_cls()

        return optim_cls(
            get_trainable_params(),
            lr=self.base_config.learning_rate,
            weight_decay=self.base_config.weight_decay,
            eps=self.base_config.optimizer_epsilon,
        )

    def get_scheduler(
        self, optimizer: Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        if self.config.unfreeze_strategy == "brain_encoder":
            # return scheduler from super
            assert (
                self.config.w2v_warmup_steps is None
            ), "w2v_warmup_steps can only be set if unfreeze strategy is brain_encoder+w2v"
            assert (
                self.config.adjust_global_lr_to_w2v_postwarmup_lr is None
            ), "adjust_global_lr_to_w2v_postwarmup_lr can only be set if unfreeze strategy is brain_encoder+w2v"
            return super().get_scheduler(optimizer)

        warmup_start_step = (
            self.config.w2v_warmup_start_step
            if self.config.w2v_warmup_start_step
            else 0
        )
        warmup_steps = (
            self.config.w2v_warmup_steps if self.config.w2v_warmup_steps else 0
        )

        return get_2module_warmup_scheduler(
            optimizer,
            self.config.learning_rate,
            warmup_start_step,
            warmup_steps,
            (
                self.config.w2v_learning_rate
                if self.config.w2v_learning_rate is not None
                else self.config.learning_rate
            ),
            self.config.adjust_global_lr_to_w2v_postwarmup_lr == True,
        )

    def create_evaluator(
        self,
        mode: Literal["train", "val", "test"],
        track_non_test_predictions: bool = False,
    ):
        return EvaluatorWithW2vLMDecoder(
            self.tokenizer,
            mode,
            self.yaml_config.cache_dir,
            W2V_CHECKPOINT_TO_PROCESSOR[self.config.wav2vec_checkpoint],
            track_non_test_predictions,
            self.config.lm_decode_test_predictions,
            self.config.lm_decode_beam_width,
            self.config.lm_decode_beam_prune_logp,
            self.config.lm_decode_token_min_logp,
            self.config.lm_decode_alpha,
            self.config.lm_decode_beta,
            self.config.lm_score_boundary,
            rerank_with_sentence_embeddings=self.config.rerank_with_sentence_embeddings,
            sentence_embedding_results_dir=self.config.sentence_embedding_results_dir,
            sentence_transformer_checkpoint=self.config.sentence_transformer_checkpoint,
            sentence_rerank_top_k=self.config.sentence_rerank_top_k,
        )
