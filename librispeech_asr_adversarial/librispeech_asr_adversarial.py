# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Adversarial examples based on the Librispeech automatic speech recognition dataset."""

from __future__ import absolute_import, division, print_function

import glob
import os

import datasets


# TODO: change
_CITATION = """\
@inproceedings{panayotov2015librispeech,
  title={Librispeech: an ASR corpus based on public domain audio books},
  author={Panayotov, Vassil and Chen,
      Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on},
  pages={5206--5210},
  year={2015},
  organization={IEEE}
}
"""

# TODO: change
_DESCRIPTION = """\
LibriSpeech is a corpus of approximately 1000 hours of read English speech with sampling rate of 16 kHz,
prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read
audiobooks from the LibriVox project, and has been carefully segmented and aligned.
Note that in order to limit the required storage for preparing this dataset, the audio
is stored in the .flac format and is not converted to a float32 array. To convert, the audio
file to a float32 array, please make use of the `.map()` function as follows:
```python
import soundfile as sf
def map_to_array(batch):
    speech_array, _ = sf.read(batch["file"])
    batch["speech"] = speech_array
    return batch
dataset = dataset.map(map_to_array, remove_columns=["file"])
```
"""
# TODO: change
_DL_URL = "https://data.mendeley.com/api/datasets/kbdt923ybx/draft/files/65e7d2bf-16f7-4eed-8ac3-e3ae5d67aa40?a=7e4a4a33-35dd-4608-9d3a-24c0de170260"

_DL_URLS = {
    "adv": {
        "test": _DL_URL,
    }
}


class LibrispeechASRConfig(datasets.BuilderConfig):
    """BuilderConfig for LibriSpeechASR."""

    def __init__(self, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        super(LibrispeechASRConfig, self).__init__(
            version=datasets.Version("2.1.0", ""), **kwargs)


class LibrispeechASR(datasets.GeneratorBasedBuilder):
    """Librispeech dataset."""

    BUILDER_CONFIGS = [
        LibrispeechASRConfig(name="adv", description="'Adversarial' speech."),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "audio": datasets.features.Audio(sampling_rate=16_000),
                    "true_text": datasets.Value("string"),
                    "target_text": datasets.Value("string"),
                    "id": datasets.Value("string"),
                }
            ),
            supervised_keys=("speech", "true_text"),
            homepage=_DL_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        archive_path = dl_manager.download_and_extract(
            _DL_URLS[self.config.name])

        return [
            datasets.SplitGenerator(name="natural", gen_kwargs={
                                    "archive_path": archive_path["test"], "split_name": f"natural"}),
            datasets.SplitGenerator(name="adv_0.04", gen_kwargs={
                                    "archive_path": archive_path["test"], "split_name": f"adv_0.04"}),
            datasets.SplitGenerator(name="adv_0.015", gen_kwargs={
                                    "archive_path": archive_path["test"], "split_name": f"adv_0.015"}),
            datasets.SplitGenerator(name="adv_0.015_RIR", gen_kwargs={
                                    "archive_path": archive_path["test"], "split_name": f"adv_0.015_RIR"}),
        ]

    def _generate_examples(self, archive_path, split_name):
        """Generate examples from a Librispeech archive_path."""
        transcript_file = os.path.join(archive_path, "manifest.txt")
        split_folder = split_name

        path = os.path.dirname(transcript_file)
        audio_path = os.path.join(path, split_folder)
        with open(transcript_file) as f:
            for line in f:
                line = line.strip()
                key, og_transcript, adv_transcript = line.split(",", 2)
                suffix = "nat" if split_folder == "natural" else "adv"
                audio_file = f"{key}_{suffix}.wav"
                split_key = key+"_"+suffix+"_"+split_name
                example = {
                    "id": split_key,
                    "file": os.path.join(audio_path, audio_file),
                    "audio": os.path.join(audio_path, audio_file),
                    "true_text": og_transcript,
                    "target_text": adv_transcript,
                }
                yield split_key, example
