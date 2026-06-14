#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
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

"""sherpa-onnx ASR helpers used by the Gradio demo and model API."""

import glob
import os
from dataclasses import dataclass
import numpy as np
import torch
import torchaudio

from omnivoice.utils.audio import load_waveform


DEFAULT_SHERPA_ASR_REPO = "csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14"
DEFAULT_SHERPA_ASR_DIR = os.path.join(
    "models_asr", "sherpa-onnx-paraformer-zh-2023-09-14"
)


@dataclass
class SherpaAsrConfig:
    model: str = DEFAULT_SHERPA_ASR_REPO
    num_threads: int = 1
    sample_rate: int = 16000
    feature_dim: int = 80
    decoding_method: str = "greedy_search"


def _resolve_paraformer_files(model: str) -> tuple[str, str]:
    if os.path.isfile(model):
        model_dir = os.path.dirname(model)
        tokens_file = os.path.join(model_dir, "tokens.txt")
        if not os.path.exists(tokens_file):
            raise FileNotFoundError(f"Cannot find tokens.txt next to {model}")
        return model, tokens_file

    if os.path.isdir(model):
        tokens_file = os.path.join(model, "tokens.txt")
        onnx_files = glob.glob(os.path.join(model, "*.onnx"))
        if not os.path.exists(tokens_file) or not onnx_files:
            raise FileNotFoundError(
                f"ASR directory must contain tokens.txt and at least one .onnx: {model}"
            )
        model_file = next((f for f in onnx_files if "int8" not in f), onnx_files[0])
        return model_file, tokens_file

    from huggingface_hub import snapshot_download

    local_dir = DEFAULT_SHERPA_ASR_DIR if model == DEFAULT_SHERPA_ASR_REPO else None
    model_dir = snapshot_download(repo_id=model, local_dir=local_dir)
    return _resolve_paraformer_files(model_dir)


def create_offline_recognizer(config: SherpaAsrConfig):
    try:
        import sherpa_onnx
    except ImportError as e:
        raise ImportError(
            "sherpa-onnx is required for automatic reference transcription. "
            "Install it with `pip install sherpa-onnx` or run the demo with --no-asr."
        ) from e

    model_file, tokens_file = _resolve_paraformer_files(config.model)
    return sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=model_file,
        tokens=tokens_file,
        num_threads=config.num_threads,
        sample_rate=config.sample_rate,
        feature_dim=config.feature_dim,
        decoding_method=config.decoding_method,
    )


def load_mono_16k(audio: str | tuple, target_sample_rate: int = 16000) -> np.ndarray:
    if isinstance(audio, str):
        waveform, sample_rate = load_waveform(audio)
    else:
        waveform, sample_rate = audio
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]

    if waveform.shape[0] > 1:
        waveform = np.mean(waveform, axis=0, keepdims=True)

    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            torch.from_numpy(waveform),
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        ).numpy()

    return np.ascontiguousarray(waveform.squeeze(0), dtype=np.float32)


def transcribe(recognizer, audio: str | tuple, sample_rate: int = 16000) -> str:
    waveform = load_mono_16k(audio, target_sample_rate=sample_rate)
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, waveform)
    recognizer.decode_stream(stream)
    text = stream.result.text.strip()
    return text.replace(' "<unk>" ', ", ").replace(' "<unk>"', ".")
