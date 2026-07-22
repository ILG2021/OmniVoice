import os
import glob
import threading

import numpy as np

import sherpa_onnx
from huggingface_hub import snapshot_download
from sherpa_onnx import OfflinePunctuation


def _find_model_files(model_dir):
    tokens_files = glob.glob(
        os.path.join(model_dir, "**", "tokens.txt"), recursive=True
    )
    onnx_files = glob.glob(
        os.path.join(model_dir, "**", "*.onnx"), recursive=True
    )
    if not tokens_files or not onnx_files:
        raise FileNotFoundError(
            f"No Paraformer tokens.txt and ONNX model found under {model_dir}"
        )
    model_file = next(
        (
            path
            for path in onnx_files
            if "int8" not in os.path.basename(path).lower()
        ),
        onnx_files[0],
    )
    return model_file, tokens_files[0]


def _init_asr_recognizer(model_name=None, num_threads=1):
    """Initialize a sherpa-onnx Paraformer recognizer."""
    try:
        asr_base_dir = "models_asr"
        model_dir = None
        model_file = None
        tokens_file = None

        if model_name:
            model_dir = (
                model_name
                if os.path.isdir(model_name)
                else snapshot_download(repo_id=model_name)
            )
            model_file, tokens_file = _find_model_files(model_dir)

        # Scan models-asr for any custom model containing tokens.txt and *.onnx.
        if model_dir is None and os.path.exists(asr_base_dir):
            for subdir in os.listdir(asr_base_dir):
                subdir_path = os.path.join(asr_base_dir, subdir)
                if os.path.isdir(subdir_path):
                    try:
                        found_model, found_tokens = _find_model_files(subdir_path)
                    except FileNotFoundError:
                        continue
                    else:
                        model_dir = subdir_path
                        tokens_file = found_tokens
                        model_file = found_model
                        print(f">> Found custom ASR model in {model_dir}, using {os.path.basename(model_file)}")
                        break

        # Fallback to default if no valid custom model found
        if model_dir is None:
            model_dir = os.path.join(asr_base_dir, "sherpa-onnx-paraformer-zh-2023-09-14")
            model_file = os.path.join(model_dir, "model.int8.onnx")
            tokens_file = os.path.join(model_dir, "tokens.txt")
            if not os.path.exists(model_file) or not os.path.exists(tokens_file):
                print(f">> Downloading ASR model to {model_dir}...")
                snapshot_download(repo_id="csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14", local_dir=model_dir)

        recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=model_file,
            tokens=tokens_file,
            num_threads=max(1, int(num_threads)),
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
        )
        print(">> ASR model initialized and resident in memory.")
        return recognizer
    except Exception as e:
        print(f">> ASR Initialization Error: {e}")
        raise RuntimeError("Failed to initialize sherpa-onnx ASR") from e

def _get_nn_model_filename(
        repo_id: str,
        filename: str,
        subfolder: str = "exp",
) -> str:
    return os.path.join(snapshot_download(repo_id),
                        filename)

def _get_punct_model() -> OfflinePunctuation:
    model = _get_nn_model_filename(
        repo_id="csukuangfj/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12",
        filename="model.onnx",
        subfolder=".",
    )
    config = sherpa_onnx.OfflinePunctuationConfig(
        model=sherpa_onnx.OfflinePunctuationModelConfig(ct_transformer=model),
    )

    punct = sherpa_onnx.OfflinePunctuation(config)
    return punct

_INIT_LOCK = threading.Lock()
_TRANSCRIBE_LOCK = threading.Lock()
_ASR_RECOGNIZER = None
PUNC_MODEL = None


def configure(model_name=None, num_threads=1):
    """Load or replace the process-wide Sherpa recognizer."""
    global _ASR_RECOGNIZER
    with _INIT_LOCK:
        _ASR_RECOGNIZER = _init_asr_recognizer(
            model_name=model_name,
            num_threads=num_threads,
        )
    return _ASR_RECOGNIZER


def _get_recognizer():
    if _ASR_RECOGNIZER is None:
        return configure()
    return _ASR_RECOGNIZER


def _get_punctuation_model():
    global PUNC_MODEL
    if PUNC_MODEL is None:
        with _INIT_LOCK:
            if PUNC_MODEL is None:
                PUNC_MODEL = _get_punct_model()
    return PUNC_MODEL

def transcribe(audio, add_punctuation=False):
    import librosa
    if audio is None:
        return ""
    recognizer = _get_recognizer()

    if isinstance(audio, tuple):
        waveform, sample_rate = audio
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().numpy()
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)
        if sample_rate != 16000:
            waveform = librosa.resample(
                waveform, orig_sr=sample_rate, target_sr=16000
            )
    else:
        waveform, sample_rate = librosa.load(audio, sr=16000, mono=True)

    with _TRANSCRIBE_LOCK:
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, waveform)
        recognizer.decode_stream(stream)
        text_result = stream.result.text
    text_result = text_result.replace("<unk>", '').strip(" ")

    if add_punctuation:
        text_result = _get_punctuation_model().add_punctuation(text_result)
    return text_result
