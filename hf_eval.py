from transformers import Wav2Vec2Processor, Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
import torchaudio

processor_lm = Wav2Vec2ProcessorWithLM.from_pretrained(
    "patrickvonplaten/wav2vec2-base-100h-with-lm")
# load model and tokenizer
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# load dummy dataset and read soundfiles
audio = torchaudio.load(
    "/home/raphael/dataspace/outputs/robust_speech/attacks/cwfinal/hubert-large-960h-wav2vec2-large-960h/10000/save/8230-279154-0036_adv.wav")[0][0].numpy()

ds = load_dataset("patrickvonplaten/librispeech_asr_dummy",
                  "clean", split="validation")
audio2 = ds[0]["audio"]["array"]
# tokenize
input_values = processor_lm(
    audio, return_tensors="pt", sampling_rate=16000).input_values  # Batch size 1

# retrieve logits
logits = model(input_values).logits
# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
transcription_lm = processor_lm.batch_decode(logits.detach().numpy()).text

print(transcription)
print(transcription_lm)
