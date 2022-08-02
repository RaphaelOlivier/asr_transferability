import sys
from transformers import Wav2Vec2Processor, Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC, HubertForCTC, Data2VecAudioForCTC, WavLMForCTC, UniSpeechSatForCTC, MCTCTForCTC, MCTCTProcessor, Speech2TextProcessor, Speech2TextForConditionalGeneration
from datasets import load_dataset
import torch
from jiwer import wer
# load dummy dataset and read soundfiles
device = "cuda" if torch.cuda.is_available() else "cpu"


def map_to_pred_wav2vec(processor, model):
    def map_to_pred(batch):
        input_values = processor(
            batch["audio"][0]["array"], return_tensors="pt", sampling_rate=16000).input_values
        with torch.no_grad():
            logits = model(input_values.to("cuda")).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        transcription = [t.upper() for t in transcription]
        batch["transcription"] = transcription
        return batch
    return map_to_pred


def map_to_pred_mctc(processor, model):
    def map_to_pred(batch):
        input_features = processor(
            batch["audio"][0]["array"], return_tensors="pt", sampling_rate=16000).input_features
        with torch.no_grad():
            logits = model(input_features.to("cuda")).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        transcription = [t.upper() for t in transcription]
        batch["transcription"] = transcription
        return batch
    return map_to_pred


def map_to_pred_s2t(processor, model):
    def map_to_pred(batch):
        input_features = processor(
            batch["audio"][0]["array"], return_tensors="pt", sampling_rate=16000).input_features
        with torch.no_grad():
            generated_ids = model.generate(input_features.to(device))
        transcription = processor.batch_decode(generated_ids)
        transcription = [t.upper() for t in transcription]
        batch["transcription"] = transcription
        return batch
    return map_to_pred


def map_to_pred_wav2vec_lm(processor, model):
    def map_to_pred(batch):
        input_values = processor(
            batch["audio"][0]["array"], return_tensors="pt", sampling_rate=16000).input_values
        with torch.no_grad():
            logits = model(input_values.to(device)).logits
        transcription = processor.batch_decode(
            logits.detach().cpu().numpy()).text
        transcription = [t.upper() for t in transcription]
        batch["transcription"] = transcription
        return batch
    return map_to_pred


MODELS = {
    "wav2vec2-base": {
        "model_class": Wav2Vec2ForCTC, "processor_class": Wav2Vec2Processor, "model_path": "facebook/wav2vec2-base-960h", "prediction_fn": map_to_pred_wav2vec, "can_use_lm": True
    },
    "wav2vec2-base-100h": {
        "model_class": Wav2Vec2ForCTC, "processor_class": Wav2Vec2Processor, "model_path": "facebook/wav2vec2-base-100h", "prediction_fn": map_to_pred_wav2vec, "can_use_lm": True
    },
    "wav2vec2-large": {
        "model_class": Wav2Vec2ForCTC, "processor_class": Wav2Vec2Processor, "model_path": "facebook/wav2vec2-large-960h", "prediction_fn": map_to_pred_wav2vec, "can_use_lm": True
    },
    "hubert-large": {
        "model_class": HubertForCTC, "processor_class": Wav2Vec2Processor, "model_path": "facebook/hubert-large-ls960-ft", "prediction_fn": map_to_pred_wav2vec, "can_use_lm": True
    },
    "hubert-xlarge": {
        "model_class": HubertForCTC, "processor_class": Wav2Vec2Processor, "model_path": "facebook/hubert-xlarge-ls960-ft", "prediction_fn": map_to_pred_wav2vec, "can_use_lm": True
    },
    "data2vec-large": {
        "model_class": Data2VecAudioForCTC, "processor_class": Wav2Vec2Processor, "model_path": "facebook/data2vec-audio-large-960h", "prediction_fn": map_to_pred_wav2vec, "can_use_lm": True
    },
    "unispeech-base": {
        "model_class": UniSpeechSatForCTC, "processor_class": Wav2Vec2Processor, "model_path": "microsoft/unispeech-sat-base-100h-libri-ft", "prediction_fn": map_to_pred_wav2vec, "can_use_lm": True
    },
    "wavlm-base-100h": {
        "model_class": WavLMForCTC, "processor_class": Wav2Vec2Processor, "model_path": "patrickvonplaten/wavlm-libri-clean-100h-base-plus", "prediction_fn": map_to_pred_wav2vec, "can_use_lm": False
    },
    "m-ctc-t-large": {
        "model_class": MCTCTForCTC, "processor_class": MCTCTProcessor, "model_path": "speechbrain/m-ctc-t-large", "prediction_fn": map_to_pred_mctc, "can_use_lm": False
    },
    "s2t-large": {
        "model_class": Speech2TextForConditionalGeneration, "processor_class": Speech2TextProcessor, "model_path": "facebook/s2t-large-librispeech-asr", "prediction_fn": map_to_pred_s2t, "can_use_lm": False
    },
}

if __name__ == "__main__":
    args = sys.argv[1:]
    model_name = args[0] if len(args) > 0 else "wav2vec2-large"
    dataset_split = args[1] if len(args) > 1 else "adv_0.015"
    lm = True if len(args) > 2 and args[2] in ["--ngram-lm", "--lm"] else False
    if model_name not in MODELS:
        raise ValueError("models available: %s" % str(MODELS.keys()))

    ds = load_dataset("librispeech_asr_adversarial",
                      "adv", split=dataset_split)

    model_path = MODELS[model_name]["model_path"]
    model = MODELS[model_name]["model_class"].from_pretrained(
        model_path).to(device)
    processor = MODELS[model_name]["processor_class"].from_pretrained(
        model_path)
    pred_fn = MODELS[model_name]["prediction_fn"](processor, model)
    if lm:
        if not MODELS[model_name]["can_use_lm"]:
            raise ValueError(
                "Model %s is not compatible with Wav2Vec2 ngram language model")
        processor = Wav2Vec2ProcessorWithLM.from_pretrained(
            "patrickvonplaten/wav2vec2-base-100h-with-lm")
        pred_fn = map_to_pred_wav2vec_lm(processor, model)
    print("Using model %s and split %s" % (model_name, dataset_split))

    result = ds.map(pred_fn, batched=True,
                    batch_size=1, remove_columns=["audio"])
    print("WER on true labels:", wer(
        result["true_text"], result["transcription"]))
    print("WER on adversarial labels:", wer(
        result["target_text"], result["transcription"]))
