import os
import glob

import sherpa_onnx
from huggingface_hub import snapshot_download
from sherpa_onnx import OfflinePunctuation


def _init_asr_recognizer():
    """模块加载时初始化 ASR recognizer，和 vocoder 一样在启动时完成，无需懒加载锁。"""
    try:
        import sherpa_onnx
        from huggingface_hub import snapshot_download

        asr_base_dir = "models_asr"
        model_dir = None
        model_file = None
        tokens_file = None

        # Scan models-asr for any custom model containing tokens.txt and *.onnx
        if os.path.exists(asr_base_dir):
            for subdir in os.listdir(asr_base_dir):
                subdir_path = os.path.join(asr_base_dir, subdir)
                if os.path.isdir(subdir_path):
                    t_file = os.path.join(subdir_path, "tokens.txt")
                    onnx_files = glob.glob(os.path.join(subdir_path, "*.onnx"))
                    if os.path.exists(t_file) and len(onnx_files) > 0:
                        model_dir = subdir_path
                        tokens_file = t_file
                        # Prioritize non-int8 model if multiple exist, else just pick the first
                        model_file = next((f for f in onnx_files if "int8" not in f), onnx_files[0])
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
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
        )
        print(">> ASR model initialized and resident in memory.")
        return recognizer
    except Exception as e:
        print(f">> ASR Initialization Error: {e}")
        return None

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

_ASR_RECOGNIZER = _init_asr_recognizer()
PUNC_MODEL = _get_punct_model()

def transcribe(audio_path):
    import librosa
    if not audio_path:
        return ""
    if _ASR_RECOGNIZER is None:
        return "asr模型加载失败"

    audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    stream = _ASR_RECOGNIZER.create_stream()
    stream.accept_waveform(16000, audio)
    _ASR_RECOGNIZER.decode_stream(stream)
    text_result = stream.result.text
    text_result = text_result.replace(' "<unk>" ', '，').replace(' "<unk>"', "。")
    # text_result = PUNC_MODEL.add_punctuation(text_result)
    return text_result
