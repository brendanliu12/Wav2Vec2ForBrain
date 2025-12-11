import os
import re
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional

import numpy as np
import torch
from scipy.io import loadmat
from torch.nn.functional import pad
from transformers import PreTrainedTokenizer
from sentence_transformers import SentenceTransformer

from src.args.base_args import B2TDatasetArgsModel
from src.args.yaml_config import YamlConfigModel
from src.datasets.base_dataset import BaseDataset, Sample
from src.datasets.batch_types import B2tSampleBatch
from src.datasets.preprocessing import (
    Area,
    preprocess_competition_recommended,
    preprocess_only_spikepow_unnormalized,
    preprocess_only_spikepow_zscored,
    preprocess_only_tx_unnormalized,
    preprocess_only_tx_zscored,
    preprocess_seperate_zscoring,
    preprocess_seperate_zscoring_2channels,
    preprocess_seperate_zscoring_4channels,
    resample_sample,
)
from src.util.nn_helper import calc_seq_len

PreprocessingFunctions: dict[
    str,
    Callable[
        [dict, list[np.ndarray[Any, np.dtype[np.int32]]], Area], tuple[list, list[str]]
    ],
] = {
    "competition_recommended": preprocess_competition_recommended,
    "seperate_zscoring": preprocess_seperate_zscoring,
    "only_tx_unnormalized": preprocess_only_tx_unnormalized,
    "only_tx_zscored": preprocess_only_tx_zscored,
    "only_spikepow_unnormalized": preprocess_only_spikepow_unnormalized,
    "only_spikepow_zscored": preprocess_only_spikepow_zscored,
    "seperate_zscoring_2channels": preprocess_seperate_zscoring_2channels,
    "seperate_zscoring_4channels": preprocess_seperate_zscoring_4channels,
}

sessionNames = [
    "t12.2022.04.28",
    "t12.2022.05.26",
    "t12.2022.06.21",
    "t12.2022.07.21",
    "t12.2022.08.13",
    "t12.2022.05.05",
    "t12.2022.06.02",
    "t12.2022.06.23",
    "t12.2022.07.27",
    "t12.2022.08.18",
    "t12.2022.05.17",
    "t12.2022.06.07",
    "t12.2022.06.28",
    "t12.2022.07.29",
    "t12.2022.08.23",
    "t12.2022.05.19",
    "t12.2022.06.14",
    "t12.2022.07.05",
    "t12.2022.08.02",
    "t12.2022.08.25",
    "t12.2022.05.24",
    "t12.2022.06.16",
    "t12.2022.07.14",
    "t12.2022.08.11",
]
sessionNames.sort()


class B2tSample(Sample):
    day_idx: int


class Brain2TextDataset(BaseDataset):
    def __init__(
        self,
        config: B2TDatasetArgsModel,
        yaml_config: YamlConfigModel,
        split: Literal["train", "val", "test"] = "train",
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.use_sentence_embeddings = getattr(config, "predict_sentence_embeddings", False)
        self.sentence_transformer = None
        if self.use_sentence_embeddings:
            #self.sentence_transformer = SentenceTransformer("all-mpnet-base-v2")
            self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")


        if split == "val":
            data_path = Path(yaml_config.dataset_splits_dir) / "test"
        elif split == "test" and config.competition_mode:
            data_path = Path(yaml_config.dataset_splits_dir) / "competitionHoldOut"
        else:
            data_path = Path(yaml_config.dataset_splits_dir) / "train"

        if not os.path.exists(data_path):
            raise Exception(f"{data_path} does not exist.")

        data_files = [
            (day_idx, loadmat(data_path / f"{filePrefix}.mat"))
            for day_idx, filePrefix in enumerate(sessionNames)
            if os.path.exists(data_path / f"{filePrefix}.mat")
        ]

        self.tokenizer = tokenizer
        preprocess = PreprocessingFunctions[config.preprocessing]

        # Samples are made up of a tuple of (day_idx, brain_data_sample, transcription)
        self.samples: list[B2tSample] = []

        for day_idx, data_file in data_files:
            # block-wise feature normalization
            blockNums = np.squeeze(data_file["blockIdx"])
            blockList = np.unique(blockNums)

            if split == "test" and not config.competition_mode:
                blockList = [blockList[0]]
            if split == "train" and not config.competition_mode:
                blockList = blockList[1:]

            blocks = []
            for b in range(len(blockList)):
                sentIdx = np.argwhere(blockNums == blockList[b])
                sentIdx = sentIdx[:, 0].astype(np.int32)
                blocks.append(sentIdx)

            input_features, transcriptions = preprocess(data_file, blocks, config.area)

            assert len(input_features) == len(
                transcriptions
            ), "Length of input features and transcriptions must be equal."

            for i in range(len(input_features)):
                x = torch.tensor(input_features[i], dtype=torch.float32)
                text = transcriptions[i].upper()

                sample = B2tSample(x, text)
                sample.day_idx = day_idx

                if self.use_sentence_embeddings and self.sentence_transformer is not None:
                    emb = self.sentence_transformer.encode(
                        text,
                        convert_to_numpy=True,
                        normalize_embeddings=True, 
                    )
                    sample.target_embedding = torch.tensor(emb, dtype=torch.float32)

                self.samples.append(sample)

    def __len__(self):
        return (
            len(self.samples)
            if self.config.limit_samples is None
            else min(len(self.samples), self.config.limit_samples)
        )

    def __getitem__(self, index: int) -> B2tSample:
        orig_sample_rate = 50
        target_sample_rate = self.config.sample_rate

        if target_sample_rate % orig_sample_rate != 0:
            print("WARNING: target_sample_rate % orig_sample_rate != 0")

        orig_sample = self.samples[index]
        brain_data = orig_sample.input
        resampled = (
            resample_sample(brain_data, target_sample_rate, orig_sample_rate)
            if target_sample_rate != orig_sample_rate
            else brain_data
        )
        resampled_sample = B2tSample(resampled, orig_sample.target)
        resampled_sample.day_idx = orig_sample.day_idx

        if hasattr(orig_sample, "target_embedding") and orig_sample.target_embedding is not None:
            resampled_sample.target_embedding = orig_sample.target_embedding

        return resampled_sample

    def get_collate_fn(
        self, tokenizer: Optional[PreTrainedTokenizer]
    ) -> Callable[[List[B2tSample]], B2tSampleBatch]:
       
        use_embeddings = getattr(self.config, "predict_sentence_embeddings", False)

        # For text/CTC experiments we still require a tokenizer
        if not use_embeddings and tokenizer is None:
            raise ValueError(
                "Tokenizer must be provided for text/CTC collate; "
                "set predict_sentence_embeddings=True to use embedding targets without a tokenizer."
            )

        multiple_channels = (
            self.config.preprocessing == "seperate_zscoring_2channels"
            or self.config.preprocessing == "seperate_zscoring_4channels"
        )

        def _collate(batch: List[B2tSample]) -> B2tSampleBatch:
            # sample.input shape: [T, F] or [C, T, F] depending on preprocessing
            if multiple_channels:
                # time dimension is axis 1
                max_block_len = max(sample.input.size(1) for sample in batch)
            else:
                # time dimension is axis 0
                max_block_len = max(sample.input.size(0) for sample in batch)

            padded_blocks = []
            for sample in batch:
                x = sample.input
                if multiple_channels:
                    # x: [C, T, F]  -> pad along T dimension
                    pad_T = max_block_len - x.size(1)
                    padded = pad(x, (0, 0, 0, pad_T, 0, 0), mode="constant", value=0)
                else:
                    # x: [T, F] -> pad along T dimension
                    pad_T = max_block_len - x.size(0)
                    padded = pad(x, (0, 0, 0, pad_T), mode="constant", value=0)
                padded_blocks.append(padded)

            inputs = torch.stack(padded_blocks)  # [B, ..., T_max, F] or [B, C, T_max, F]

            # Length along time dimension (always size(0) in your original code)
            input_lens = torch.tensor([sample.input.size(0) for sample in batch], dtype=torch.long)

            day_idxs = torch.tensor([sample.day_idx for sample in batch], dtype=torch.long)
            if use_embeddings:
                # Each sample must have target_embedding set
                target_embs = torch.stack(
                    [sample.target_embedding for sample in batch]  # [B, d]
                )

                collated_batch = B2tSampleBatch(
                    input=inputs,
                    target=None,            
                    target_embedding=target_embs,
                )
                collated_batch.input_lens = input_lens
                collated_batch.day_idxs = day_idxs
                return collated_batch


            def process_label(label: str) -> str:
                if self.config.remove_punctuation:
                    chars_to_ignore_regex = r'[\,\?\.\!\-\;\:"]'
                    label = re.sub(chars_to_ignore_regex, "", label)
                # label = label.upper()
                return label

            texts = [process_label(sample.target) for sample in batch]

            # text branch
            batch_label_ids: torch.Tensor = tokenizer(
                texts,
                padding="longest",
                return_tensors="pt",
            ).input_ids

            collated_batch = B2tSampleBatch(
                input=inputs,
                target=batch_label_ids,
            )
            collated_batch.day_idxs = day_idxs
            collated_batch.input_lens = input_lens
            collated_batch.target_lens = torch.tensor(
                [calc_seq_len(label_ids) for label_ids in batch_label_ids],
                dtype=torch.long,
            )
            return collated_batch

        return _collate
