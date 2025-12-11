# Brain2Text
This project was started in the context of Competitive Programming with Deep Learning Seminar at HPI. 
We investigate if knowledge transfer from Wav2Vec2 (speech recognition from audio) to the task of brain to text decoding is possible.
[arXiv preprint](https://arxiv.org/html/2501.09459v1)

## Setup
1. Create conda env `conda env create -f environment.yaml`
2. Activate env: `conda activate b2t`
3. Set Python path to repository root dir: `export PYTHONPATH="[...]/brain2text"`
2. Run `python run.py --experiment_type=b2p2t_gru+w2v`, it will create a `config.yaml` in project root and prompt you to specify cache directories, API keys etc. before running again

### Update conda environment from environment.yaml
```
conda activate b2t
conda env update --file environment.yaml --prune
```

#### If Update leads to cudaKernelExc Error
```
conda create --name b2t pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda activate b2t
conda install scipy pydantic tokenizers transformers matplotlib
conda install -c conda-forge wandb
``` 

## Execute experiments
To run the experiments for 45 Brain Feature Extractor Architectures each for our three experiment setups (we used a NVIDIA A100 GPU and 32GB of RAM):
1. Start WandB sweep for corresponding experiment setup: `wandb sweep sweeps/[...experiment setup sweep file].yaml`
2. Connect one or more agents to execute the runs `wandb agent [SWEEP ID]`

The results of our runs can be found in the [analysis dir](./src/analysis/data).
If you set `--lm_decode_test_predictions=true` when executing `run.py`, you probably need 64GB of RAM (at least for batch size 64).

To run the Wav2Vec2Conformer experiment to reproduce our results, execute `python run.py --encoder_fc_hidden_sizes=[256] --encoder_gru_hidden_size=512 --encoder_num_gru_layers=3 --use_wandb=true --experiment_type=b2p2t_gru+w2v_conformer --loss_function=ctc --early_stopping_patience=10 --epochs=100 --batch_size=32 --learning_rate=0.0001 --return_best_model=false --encoder_learnable_inital_state=false --unfreeze_strategy=brain_encoder+w2v --weight_decay=8.324385138271928e-05 --encoder_dropout=0.4570249990196249 --gaussian_smooth_width=1.5290517142639226 --w2v_learning_rate=9.506050391898906e-06 --w2v_warmup_steps=7 --w2v_warmup_start_step=7 --whiteNoiseSD=0.01978441712172472 --constantOffsetSD=0.2443028255597108 --lm_decode_test_predictions=true --wav2vec_checkpoint=facebook/wav2vec2-conformer-rope-large-960h-ft --tokenizer_checkpoint=facebook/wav2vec2-conformer-rope-large-960h-ft --experiment_name="gru+w2v conformer large"`

To run embeddings experiment, execute 
python run.py  --experiment_type=b2p2t_44_sentence_embedding  --area=44 --predict_sentence_embeddings=true --batch_size=32  --epochs=100 --learning_rate=0.0001  --use_wandb=false  --experiment_name="area44_sentence_embeddings_all-MiniLM-L6-v2" --wav2vec_checkpoint=facebook/wav2vec2-base-960h
