# asr_transferability

This code folder accompanies the [paper](https://arxiv.org/abs/2209.13523) "Watch What You Pretrain For: Targeted, Transferable Adversarial Examples on Self-Supervised Speech Recognition models".

## Setup
The only requirements are to install [robust_speech](https://github.com/RaphaelOlivier/robust_speech) and [fairseq](https://github.com/facebookresearch/fairseq)

## Use

In order to use our adversarial dataset on any huggingface transformer model, you should use the `hf_eval.py` script (see specific README in the `librispeech_asr_adversarial` folder).

To reproduce our experiments, we provide all the robust_speech attack, training and model configuration files that we used. The training script `rs_train.py` and training configurations are mostly used to download HuggingFace models fine-tune their projection layer onto the same vocabulary. Once trained you can use the `evaluate.py` file to run adversarial attacks. The `cw.yaml` configuration file will run a simple attack on a single model. `cw_tgt.yaml` runs an attack on a model while using another model as a stopping criterion. `cw_tgt_multi.yaml` extends the attack by optimizing the loss on two different models. For instance to generate an adversarial dataset similar to ours you could run 

`python evaluate.py cw_tgt_multi.yaml --root=/path/to/robust_speech/root/folder  --source_model_name1=hubert-large-960h --source_model_name2=wav2vec2-large-960h-lv60-self --source_model_name3=data2vec-base-960h --save_audio=True`

A few additional scripts are also provided. `fs_train.py` can be used to train wav2vec2 models from scratch using our two fairseq configuration files. `cifar10.py` reproduces the *very targeted attack on images* experiment from the paper. The `prefix_pgd.yaml` configuration file can be used along with `evaluate.py` to reproduce the *mildly targeted attack on ASR* experiment
