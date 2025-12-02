import os
import re
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch
from scipy.io import loadmat
from torch.nn.functional import pad
from transformers import PreTrainedTokenizer

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

        # -----------------------------
        # Choose the data path by split
        # -----------------------------
        if split == "val":
            data_path = Path(yaml_config.dataset_splits_dir) / "test"
        elif split == "test" and config.competition_mode:
            data_path = Path(yaml_config.dataset_splits_dir) / "competitionHoldOut"
        else:
            # train or non-competition test
            data_path = Path(yaml_config.dataset_splits_dir) / "train"

        if not os.path.exists(data_path):
            raise Exception(f"{data_path} does not exist.")

        # -----------------------------
        # Load all .mat files
        # -----------------------------
        data_files = [
            (day_idx, loadmat(data_path / f"{filePrefix}.mat"))
            for day_idx, filePrefix in enumerate(sessionNames)
            if os.path.exists(data_path / f"{filePrefix}.mat")
        ]

        self.tokenizer = tokenizer
        preprocess = PreprocessingFunctions[config.preprocessing]

        # Samples are made up of a tuple of (day_idx, brain_data_sample, transcription)
        self.samples: list[B2tSample] = []

        # -----------------------------
        # Build samples
        # -----------------------------
        for day_idx, data_file in data_files:
            # block-wise feature normalization
            blockNums = np.squeeze(data_file["blockIdx"])
            blockList = np.unique(blockNums)

            # train/val/test splitting by block
            if split == "test" and not config.competition_mode:
                blockList = [blockList[0]]
            if split == "train" and not config.competition_mode:
                blockList = blockList[1:]

            blocks = []
            for b in range(len(blockList)):
                sentIdx = np.argwhere(blockNums == blockList[b])
                sentIdx = sentIdx[:, 0].astype(np.int32)
                blocks.append(sentIdx)

            # -----------------------------
            # Main features: current area (e.g. "6v")
            # -----------------------------
            input_features, transcriptions = preprocess(
                data_file,
                blocks,
                config.area,
            )

            assert len(input_features) == len(
                transcriptions
            ), "Length of input features and transcriptions must be equal."

            # -----------------------------
            # Optional: also compute area 44
            # -----------------------------
            if getattr(config, "use_area44_sentence_bias", False):
                # Force area="44" here regardless of config.area
                input_features_44, _ = preprocess(
                    data_file,
                    blocks,
                    area="44",
                )
                assert len(input_features_44) == len(
                    input_features
                ), "Area-44 and main features must have same length."
            else:
                input_features_44 = None

            # -----------------------------
            # Create B2tSample objects
            # -----------------------------
            for i in range(len(input_features)):
                sample = B2tSample(
                    torch.tensor(input_features[i], dtype=torch.float32),
                    transcriptions[i].upper(),
                )
                sample.day_idx = day_idx

                if input_features_44 is not None:
                    sample.input_44 = torch.tensor(
                        input_features_44[i], dtype=torch.float32
                    )

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

        # ----- main brain data (e.g. 6v) -----
        brain_data = orig_sample.input  # [T, F]
        resampled = (
            resample_sample(brain_data, target_sample_rate, orig_sample_rate)
            if target_sample_rate != orig_sample_rate
            else brain_data
        )

        resampled_sample = B2tSample(resampled, orig_sample.target)
        resampled_sample.day_idx = orig_sample.day_idx

        # ----- optional area-44 data -----
        if hasattr(orig_sample, "input_44"):
            brain_data_44 = orig_sample.input_44  # [T_44, F_44]
            resampled_44 = (
                resample_sample(brain_data_44, target_sample_rate, orig_sample_rate)
                if target_sample_rate != orig_sample_rate
                else brain_data_44
            )
            resampled_sample.input_44 = resampled_44

        return resampled_sample


    def get_collate_fn(
        self, tokenizer: Optional[PreTrainedTokenizer]
    ) -> Callable[[list[B2tSample]], B2tSampleBatch]:
        if tokenizer is None:
            raise ValueError(
                "Tokenizer must be provided for this implementation of collate function."
            )
        multiple_channels = (
            self.config.preprocessing == "seperate_zscoring_2channels"
            or self.config.preprocessing == "seperate_zscoring_4channels"
        )

        def _collate(batch: list[B2tSample]):
            # main input (e.g. 6v)
            max_block_len = max(
                [sample.input.size(1 if multiple_channels else 0) for sample in batch]
            )
            padded_blocks = [
                pad(
                    sample.input,
                    (
                        0,
                        0,
                        0,
                        max_block_len
                        - sample.input.size(1 if multiple_channels else 0),
                    ),
                    mode="constant",
                    value=0,
                )
                for sample in batch
            ]

            def process_label(label: str) -> str:
                if self.config.remove_punctuation:
                    chars_to_ignore_regex = r'[\,\?\.\!\-\;\:"]'
                    label = re.sub(chars_to_ignore_regex, "", label)
                return label.upper()

            # labels are still read from the tuple (input, label) interface
            batch_label_ids: torch.Tensor = tokenizer(
                [process_label(label) for _, label in batch],
                padding="longest",
                return_tensors="pt",
            ).input_ids

            collated_batch = B2tSampleBatch(
                torch.stack(padded_blocks), batch_label_ids
            )
            collated_batch.day_idxs = torch.tensor(
                [sample.day_idx for sample in batch]
            )
            collated_batch.input_lens = torch.tensor(
                [sample.input.size(0) for sample in batch]
            )
            collated_batch.target_lens = torch.tensor(
                [calc_seq_len(label_ids) for label_ids in batch_label_ids]
            )

            # If we’re using area-44 sentence bias, also collate 44 inputs
            if getattr(self.config, "use_area44_sentence_bias", False):
                # All samples should have input_44
                assert all(
                    hasattr(sample, "input_44") for sample in batch
                ), "use_area44_sentence_bias=True but some samples lack input_44"

                max_block_len_44 = max(
                    [sample.input_44.size(0) for sample in batch]
                )
                padded_blocks_44 = [
                    pad(
                        sample.input_44,
                        (0, 0, 0, max_block_len_44 - sample.input_44.size(0)),
                        mode="constant",
                        value=0,
                    )
                    for sample in batch
                ]

                collated_batch.input_44 = torch.stack(padded_blocks_44)
                collated_batch.input_44_lens = torch.tensor(
                    [sample.input_44.size(0) for sample in batch]
                )

            return collated_batch


        return _collate
