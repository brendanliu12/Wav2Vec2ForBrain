from typing import Literal, Optional

from pydantic import BaseModel, Field

PRETRAINED_LATENT_SIZES = {
    "jonatasgrosman/wav2vec2-large-xlsr-53-english": 1024,
    "facebook/wav2vec2-base-960h": 768,
    "facebook/wav2vec2-large-960h": 1024,
    "facebook/wav2vec2-conformer-rope-large-960h-ft": 1024,
    "facebook/wav2vec2-lv-60-espeak-cv-ft": 1024,
}


class B2TDatasetArgsModel(BaseModel):
    preprocessing: Literal[
        "competition_recommended",
        "seperate_zscoring",
        "only_tx_unnormalized",
        "only_tx_zscored",
        "only_spikepow_unnormalized",
        "only_spikepow_zscored",
        "seperate_zscoring_2channels",
        "seperate_zscoring_4channels",
    ] = "seperate_zscoring"
    competition_mode: bool = False
    limit_samples: Optional[int] = Field(
        default=None, description="Limit number of samples"
    )
    sample_rate: int = 50
    remove_punctuation: bool = True
    area: Literal["6v", "44"] = "6v"
    use_area44_sentence_bias: bool = False


class CTCTextDatasetArgsModel(BaseModel):
    limit_samples: Optional[int] = Field(None, description="Limit number of samples")
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    avg_num_blank_after_char: int = 6
    insert_wrong_char_prob: float = 0.05
    remove_char_prob: float = 0.05
    noise_mean: float = -16
    noise_std: float = 2
    correct_as_second_prob: float = 0.2
    random_second_id_in_blank_prob: float = 0.1
    cache_generated_samples: bool = False
    remove_punctuation: bool = True


class BaseExperimentArgsModel(BaseModel):
    batch_size: int = Field(16, description="Batch size for training and validation")
    epochs: int = 10
    learning_rate: float = 0.001
    optimizer: Literal["adam", "sgd"] = "adam"
    loss_function: Literal[
        "ctc",
        "contrastive_loss",
        "cross_entropy",
        "bce",
        "ctc+discriminator",
        "combined_ctc",
    ] = "ctc"
    ctc_loss_reduction: Literal["sum", "mean"] = "mean"
    experiment_name: str = "experiment_1"
    experiment_type: Literal[
        "audio_wav2vec2",
        "b2t_audio_wav2vec",
        "onehot_index",
        "b2t_cnn",
        "b2t_gru",
        "b2t_gru+trafo",
        "mvts_transformer",
        "b2t_mamba",
        "ctc_lm",
        "b2t_ctc_lm_mamba_finetuning",
        "b2p2t_mamba",
        "b2p2t_gru",
        "b2p2t_mvtst",
        "timit_w2v_suc",
        "timit_w2v_suc_ctc",
        "b2p_suc",
        "discriminator",
        "b2p2t_gru+w2v",
        "b2p2t_phonemegru+w2v",
        "b2p2t_gru+w2v_conformer",
        "b2p2t_gru_w2vphoneme",
        "a2p_w2vphoneme_head",
        "a2t_w2vphoneme_head",
        "b2p_w2vphoneme_head",
    ] = Field("b2t_wav2vec_sharedaggregation")
    log_every_n_batches: int = 10
    scheduler: Literal["step"] = "step"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    return_best_model: bool = True
    best_model_metric: str = Field(
        "loss",
        description='The metric by which to measure the models performance. Can be "loss" for using the applied loss or any metric that is returned by the model',
    )
    minimize_best_model_metric: bool = Field(
        True,
        description="Specify if best_model_metric should be minimized or maximized",
    )
    use_wandb: bool = False
    from_checkpoint: Optional[str] = Field(
        None, description="(optional) Path to model checkpoint"
    )
    only_test: bool = Field(False, description="Only run test, skip training")
    predict_on_train: bool = Field(
        False, description="Run prediction on train set after model training"
    )
    gradient_clipping: Optional[float] = None
    weight_decay: float = 0.0
    visualize_predictions_n_batches: int = 1
    use_fast_tokenizer: bool = False
    use_prefix_beam_search: bool = True
    beam_search_language_model: str = "openai-community/gpt2"
    whiteNoiseSD: float = 0.0
    constantOffsetSD: float = 0.0
    seed: int = 42
    optimizer_epsilon: float = 1e-8
    early_stopping_patience: Optional[int] = Field(
        None,
        description="Number of epochs n to consider for early stopping. Once all n-1 last epochs did not improve compared to the -nth epoch, training is stopped.   If None, early stopping is disabled",
    )
    early_stopping_delta: float = Field(
        0.0001,
        description="Minimum delta of to be optimized metric that is considered as an improvement for early stopping",
    )
    train_on_val_once: bool = Field(
        False, description="Train once on val after normal training"
    )
    log_results_as_artifact: bool = False
    results_subdir_name: Optional[str] = None
