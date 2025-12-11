from typing import Dict

from src.experiments.b2t_experiment import B2TExperiment
from src.experiments.b2t_gru_w2v_experiment import B2TGruAndW2VArgsModel
from src.datasets.brain2text import Brain2TextDataset
from src.model.brain_feature_extractor import bfe_w_preprocessing_from_config
from src.model.brain_to_sentence_embedding import BrainToSentenceEmbeddingModel
from src.args.yaml_config import YamlConfigModel


class B2T44SentenceEmbeddingExperiment(B2TExperiment):
    @staticmethod
    def get_args_model():
        # 🔹 Use the same args model as GRU+W2V
        # This includes:
        #   - B2TArgsModel  (dataset + base exp args)
        #   - B2P2TBrainFeatureExtractorArgsModel (encoder_* params, unfolder_kernel_len, etc.)
        #   - W2VBrainEncoderModelArgs (wav2vec_checkpoint, etc.)
        #   - brain_encoder_path: Optional[str]
        return B2TGruAndW2VArgsModel

    def __init__(self, config: Dict, yamlConfig: YamlConfigModel):
        # Force area=44 and sentence-embedding mode BEFORE parent init
        config["area"] = "44"
        config["predict_sentence_embeddings"] = True

        super().__init__(config, yamlConfig)

    def get_name(self) -> str:
        # Used for logging / dir naming
        return "b2p2t_44_sentence_embedding"

    def _create_model(self):
        # 🔹 Build the brain encoder with B2P2T + brain feature extractor
        #    Now self.config has:
        #       - encoder_rnn_hidden_size, encoder_num_gru_layers, unfolder_kernel_len, ...
        #       - wav2vec_checkpoint (needed to size the latent dimension)
        #       - brain_encoder_path (optional path to pre-trained weights)
        brain_encoder = bfe_w_preprocessing_from_config(
            self.config,
            self.config.brain_encoder_path,       # can be None if you don't pass it
            self.config.wav2vec_checkpoint,
        )

        # 🔹 Infer the embedding dimension from one training sample
        train_ds: Brain2TextDataset = self.dataloader_train.dataset  # type: ignore[attr-defined]
        sample0 = train_ds[0]
        emb_dim = sample0.target_embedding.shape[-1]

        model = BrainToSentenceEmbeddingModel(
            brain_encoder=brain_encoder,
            emb_dim=emb_dim,
            use_cosine_loss=True,
        )
        return model.cuda()
