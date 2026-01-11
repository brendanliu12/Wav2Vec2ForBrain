from torch.optim.optimizer import Optimizer
from src.datasets.batch_types import SampleBatch
from src.model.b2tmodel import ModelOutput
from src.args.base_args import B2TDatasetArgsModel, BaseExperimentArgsModel
from src.datasets.brain2text import Brain2TextDataset
from src.experiments.experiment import Experiment
from src.args.yaml_config import YamlConfigModel
from typing import Any, Literal, cast
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader
from src.train.evaluator import DefaultEvaluator
from src.train.history import DecodedPredictionBatch
from src.util.batch_sampler import Brain2TextBatchSampler
import torch


class B2TArgsModel(BaseExperimentArgsModel, B2TDatasetArgsModel):
    tokenizer: Literal["wav2vec_pretrained", "ours"] = "wav2vec_pretrained"
    tokenizer_checkpoint: str = "facebook/wav2vec2-base-100h"
    day_batches: bool = False


class B2TExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        self.yaml_config = yamlConfig
        self.tokenizer = cast(PreTrainedTokenizer, self._create_tokenizer())
        super().__init__(config, yamlConfig)

    def get_name(self) -> str:
        raise NotImplementedError()

    @staticmethod
    def get_args_model():
        return B2TArgsModel

    def _create_tokenizer(self):
        if self.config.tokenizer == "wav2vec_pretrained":
            assert (
                not self.config.tokenizer_checkpoint is None
            ), "Tokenizer checkpoint (--tokenizer_checkpoint) must be set when using --tokenizer=wav2vec_pretrained"

            return AutoTokenizer.from_pretrained(
                self.config.tokenizer_checkpoint,
                cache_dir=self.yaml_config.cache_dir,
                use_fast=self.config.use_fast_tokenizer,
            )
        raise Exception(f"Tokenizer {self.config.tokenizer} not supported yet")

    def _create_model(self):
        raise NotImplementedError()

    def decode_predictions(
        self, predictions: ModelOutput, sample: SampleBatch
    ) -> DecodedPredictionBatch:
        predicted_ids = predictions.logits.argmax(dim=-1).cpu().numpy()
        predicted_strings = self.tokenizer.batch_decode(
            predicted_ids, group_tokens=True
        )
        label_strings = (
            self.tokenizer.batch_decode(sample.target.cpu().numpy(), group_tokens=False)
            if sample.target is not None
            else None
        )
        return DecodedPredictionBatch(predicted_strings, label_strings)


    def create_optimizer(self):
        """
        Two-group LR:
        - encoder (brain_encoder): base LR (self.config.learning_rate)
        - projection head (proj): higher LR (head_lr_multiplier * base LR)
        """
        base_lr = float(self.config.learning_rate)
        head_lr_multiplier = 3.0   # try 10x first (e.g., 3e-6 -> 3e-5)
        head_lr = base_lr * head_lr_multiplier

        model = self.model

        # These attribute names match BrainToSentenceEmbeddingModel
        if hasattr(model, "brain_encoder") and hasattr(model, "proj"):
            encoder_params = list(model.brain_encoder.parameters())
            head_params = list(model.proj.parameters())

            # Optional safety: ensure we didn’t miss other params
            # (rare, but helps if you add modules later)
            encoder_ids = {id(p) for p in encoder_params}
            head_ids = {id(p) for p in head_params}
            other_params = [p for p in model.parameters() if id(p) not in encoder_ids | head_ids]

            param_groups = [
                {"params": encoder_params, "lr": base_lr},
                {"params": head_params, "lr": head_lr},
            ]
            if other_params:
                # default any "other" parameters to base_lr
                param_groups.append({"params": other_params, "lr": base_lr})

            print(f"[optimizer] encoder lr={base_lr:.2e}, head lr={head_lr:.2e}, other lr={base_lr:.2e}")

            return torch.optim.Adam(param_groups, weight_decay=1e-4)

        # Fallback for other experiment/model types
        print(f"[optimizer] single lr={base_lr:.2e} (no brain_encoder/proj attrs found)")
        return torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-4)


    def _create_dataset(self, split: Literal["train", "val", "test"] = "train"):
        return Brain2TextDataset(
            config=self.config,
            yaml_config=self.yaml_config,
            split=split,
            tokenizer=self.tokenizer,
        )

    def _create_dataloader(self, split: Literal["train", "val", "test"]) -> DataLoader:
        ds = self._create_dataset(split)

        if self.config.day_batches and split == "train":
            batch_sampler = Brain2TextBatchSampler(ds, self.base_config.batch_size)

            return DataLoader(
                self._create_dataset(split),
                batch_sampler=batch_sampler,
                collate_fn=ds.get_collate_fn(self.tokenizer),
            )
        else:
            return DataLoader(
                self._create_dataset(split),
                batch_size=self.base_config.batch_size,
                shuffle=split == "train",
                collate_fn=ds.get_collate_fn(self.tokenizer),
            )

    def get_vocab(self) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(
            list(range(self.tokenizer.vocab_size))
        )

    def create_evaluator(
        self,
        mode: Literal["train", "val", "test"],
        track_non_test_predictions: bool = False,
    ):
        return DefaultEvaluator(self.tokenizer, mode, track_non_test_predictions)
