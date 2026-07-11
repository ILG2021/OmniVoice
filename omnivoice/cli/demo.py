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
"""
Gradio demo for OmniVoice.
Supports voice cloning and voice design.

Usage:
    omnivoice-demo --model /path/to/checkpoint --port 8000
"""

import shutil
import time
import argparse
import collections
import gc
import logging
import os
import threading
from typing import Any, Callable, Dict, List

# Set Gradio temp directory to a local folder to avoid tmpfs RAM consumption on Linux
if "GRADIO_TEMP_DIR" not in os.environ:
    os.environ["GRADIO_TEMP_DIR"] = os.path.abspath("gradio_tmp")
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*HTTP_422_UNPROCESSABLE_ENTITY.*",
)

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
from omnivoice.utils import asr_sherpaonnx
from omnivoice.utils.audio import cross_fade_chunks

# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_AUTO_LABEL = "自动"
_ALL_LANGUAGES = [_AUTO_LABEL] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# Voice Design instruction templates
# ---------------------------------------------------------------------------
_CATEGORIES = {
    "性别": [("男", "Male"), ("女", "Female")],
    "年龄": [
        ("儿童", "Child"),
        ("少年", "Teenager"),
        ("青年", "Young Adult"),
        ("中年", "Middle-aged"),
        ("老年", "Elderly"),
    ],
    "音调": [
        ("极低音调", "Very Low Pitch"),
        ("低音调", "Low Pitch"),
        ("中等音调", "Moderate Pitch"),
        ("高音调", "High Pitch"),
        ("极高音调", "Very High Pitch"),
    ],
    "风格": [("耳语", "Whisper")],
    "英文口音": [
        ("美式口音", "American Accent"),
        ("澳大利亚口音", "Australian Accent"),
        ("英式口音", "British Accent"),
        ("中国口音", "Chinese Accent"),
        ("加拿大口音", "Canadian Accent"),
        ("印度口音", "Indian Accent"),
        ("韩国口音", "Korean Accent"),
        ("葡萄牙口音", "Portuguese Accent"),
        ("俄罗斯口音", "Russian Accent"),
        ("日本口音", "Japanese Accent"),
    ],
    "中文方言": [
        ("河南话", "河南话"),
        ("陕西话", "陕西话"),
        ("四川话", "四川话"),
        ("贵州话", "贵州话"),
        ("云南话", "云南话"),
        ("桂林话", "桂林话"),
        ("济南话", "济南话"),
        ("石家庄话", "石家庄话"),
        ("甘肃话", "甘肃话"),
        ("宁夏话", "宁夏话"),
        ("青岛话", "青岛话"),
        ("东北话", "东北话"),
    ],
}

_ATTR_INFO = {
    "英文口音": "仅对英文语音生效。",
    "中文方言": "仅对中文语音生效。",
}


# ---------------------------------------------------------------------------
# Model discovery and LRU cache
# ---------------------------------------------------------------------------


def _cleanup_torch_cache(device=None):
    try:
        gc.collect()
        device_str = str(device) if device is not None else ""
        if device_str.startswith("cuda") and torch.cuda.is_available():
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        elif device_str == "" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif device_str.startswith("xpu") and hasattr(torch, "xpu"):
            torch.xpu.empty_cache()
        
        # Force glibc to release free memory back to the OS on Linux
        if os.name == "posix":
            try:
                import ctypes
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except Exception:
                pass
    except Exception:
        logging.debug("Device cache cleanup failed.", exc_info=True)


def _move_model(model: OmniVoice, device):
    model.to(device)
    return model


class ModelCache:
    """LRU cache that keeps a bounded number of models on the inference device."""

    def __init__(
        self,
        *,
        device,
        gpu_slots: int,
        load_fn: Callable[[str], OmniVoice],
    ):
        self.device = device
        self.gpu_slots = max(1, int(gpu_slots))
        self.load_fn = load_fn
        self._cond = threading.Condition()
        self._gpu: dict[str, OmniVoice] = {}
        self._cpu: dict[str, OmniVoice] = {}
        self._gpu_lru: collections.OrderedDict[str, None] = collections.OrderedDict()
        self._busy_counts: collections.Counter[str] = collections.Counter()

    @staticmethod
    def cache_key(model_id: str) -> str:
        if os.path.exists(model_id):
            return os.path.normcase(os.path.realpath(os.path.abspath(model_id)))
        return model_id

    @staticmethod
    def display_name(model_id: str) -> str:
        if os.path.exists(model_id):
            try:
                return os.path.relpath(model_id, os.getcwd())
            except ValueError:
                return model_id
        return model_id

    def _log_status(self, event: str):
        gpu_names = [self.display_name(p) for p in self._gpu]
        cpu_names = [self.display_name(p) for p in self._cpu]
        logging.info(
            "[ModelCache] %s | GPU(%d/%d): %s | CPU: %s",
            event,
            len(gpu_names),
            self.gpu_slots,
            gpu_names,
            cpu_names,
        )

    def acquire(self, model_id: str) -> OmniVoice:
        key = self.cache_key(model_id)
        with self._cond:
            if key in self._gpu:
                self._busy_counts[key] += 1
                self._gpu_lru.move_to_end(key)
                self._log_status(f"reuse GPU {self.display_name(key)}")
                return self._gpu[key]

            while len(self._gpu) >= self.gpu_slots:
                evicted = next(
                    (p for p in self._gpu_lru if self._busy_counts[p] <= 0),
                    None,
                )
                if evicted is None:
                    self._cond.wait()
                    continue
                model = self._gpu.pop(evicted)
                del self._gpu_lru[evicted]
                _move_model(model, "cpu")
                self._cpu[evicted] = model
                _cleanup_torch_cache(self.device)
                self._log_status(f"GPU -> CPU {self.display_name(evicted)}")

            if key in self._cpu:
                model = self._cpu.pop(key)
                _move_model(model, self.device)
                self._log_status(f"CPU -> GPU {self.display_name(key)}")
            else:
                model = self.load_fn(key)
                self._log_status(f"load {self.display_name(key)}")

            self._gpu[key] = model
            self._gpu_lru[key] = None
            self._busy_counts[key] += 1
            return model

    def release(self, model_id: str):
        key = self.cache_key(model_id)
        with self._cond:
            if self._busy_counts[key] > 1:
                self._busy_counts[key] -= 1
            else:
                self._busy_counts.pop(key, None)
            self._cond.notify_all()


def _is_model_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))


def discover_models(model_root: str, fallback_model: str) -> dict[str, str]:
    models: dict[str, str] = {}
    if os.path.isdir(model_root):
        for name in sorted(os.listdir(model_root)):
            path = os.path.join(model_root, name)
            if _is_model_dir(path):
                models[name] = path
    if not models:
        models[ModelCache.display_name(fallback_model)] = fallback_model
    return models


def _safe_filename_part(value: str) -> str:
    value = (value or "unknown").strip()
    for ch in '<>:"/\\|?*':
        value = value.replace(ch, "_")
    return value.strip(" .") or "unknown"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="默认",
        help="Fallback model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--model-root",
        default="models",
        help="Directory containing local OmniVoice model folders (default: models).",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use. Auto-detected if not specified."
    )
    parser.add_argument("--ip", default="127.0.0.1", help="Server IP (default: 0.0.0.0).")
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)."
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="Create public link."
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading ASR model. Reference text auto-transcription"
        " will be unavailable.",
    )
    parser.add_argument(
        "--asr-backend",
        choices=["sherpa", "whisper"],
        default="sherpa",
        help="ASR backend for reference audio transcription (default: sherpa).",
    )
    parser.add_argument(
        "--asr-model",
        default=None,
        help=(
            "ASR model path or HuggingFace repo id. Defaults to "
            "csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14 for sherpa, "
            "or openai/whisper-large-v3-turbo for whisper."
        ),
    )
    parser.add_argument(
        "--asr-threads",
        type=int,
        default=8,
        help="CPU threads used by sherpa-onnx ASR (default: 8).",
    )
    parser.add_argument(
        "--concurrency",
        "-C",
        type=int,
        default=1,
        help=(
            "Max concurrent GPU inference requests and max models kept on "
            "the inference device by the LRU cache (default: 1)."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------

last_cleanup_times = {}
cleanup_lock = threading.Lock()


def delete_old_files_and_dirs(path, days=2):
    global last_cleanup_times
    now = time.time()
    with cleanup_lock:
        last_time = last_cleanup_times.get(path, 0)
        if (now - last_time) < 86400:
            return

        last_cleanup_times[path] = now

    if not os.path.exists(path):
        return

    cutoff = now - (days * 86400)
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        try:
            mtime = os.path.getmtime(full_path)
            if mtime < cutoff:
                if os.path.isfile(full_path) or os.path.islink(full_path):
                    os.remove(full_path)
                elif os.path.isdir(full_path):
                    shutil.rmtree(full_path)
        except Exception as e:
            print(f"[{path}] 处理时出错: {full_path}, 错误: {e}")

def build_demo(
    model_cache: ModelCache,
    model_choices: dict[str, str],
    generate_fn=None,
    concurrency_limit: int = 1,
) -> gr.Blocks:

    infer_semaphore = threading.BoundedSemaphore(max(1, int(concurrency_limit)))
    default_model_name = next(iter(model_choices))

    # -- shared generation core --
    def _gen_core(
        model_name,
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        ref_text=None,
    ):
        delete_old_files_and_dirs("./gen_audio", days=2)
        delete_old_files_and_dirs("./last_audio", days=2)
        delete_old_files_and_dirs("./tmp", days=2)
        delete_old_files_and_dirs("./gradio_tmp", days=2)

        model_id = model_choices.get(model_name)
        if model_id is None:
            logging.warning("[推理] 未知模型：%s", model_name)
            return None, f"未知模型：{model_name}", None
        if not text or not text.strip():
            return None, "请输入要合成的文本。", None

        # ---- 记录推理请求 ----
        logging.info(
            "[推理开始] 模型=%s | 语言=%s | 文本(前200字)=%s | "
            "参考音频=%s | 参考文本(前100字)=%s | instruct=%s | "
            "steps=%s cfg=%s speed=%s duration=%s denoise=%s pp=%s po=%s",
            model_name,
            language,
            (text or "").strip()[:200],
            ref_audio,
            (ref_text or "")[:100],
            instruct,
            num_step, guidance_scale, speed, duration,
            denoise, preprocess_prompt, postprocess_output,
        )

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != _AUTO_LABEL) else None

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        acquired = False
        _t0 = time.time()
        try:
            with infer_semaphore, torch.inference_mode():
                model = model_cache.acquire(model_id)
                acquired = True
                if ref_audio:
                    kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        preprocess_prompt=preprocess_prompt,
                    )

                if instruct and instruct.strip():
                    kw["instruct"] = instruct.strip()

                audio = model.generate(**kw)
        except Exception as e:
            logging.error(
                "[推理失败] 模型=%s 文本(前200字)=%s 参考音频=%s 异常: %s",
                model_name,
                (text or "").strip()[:200],
                ref_audio,
                e,
                exc_info=True,
            )
            raise gr.Error(f"{type(e).__name__}: {e}")
        finally:
            if acquired:
                model_cache.release(model_id)
            _cleanup_torch_cache(model_cache.device)

        output_sampling_rate = 48000
        waveform_float = audio[0]
        if model.sampling_rate != output_sampling_rate:
            waveform_float = (
                torchaudio.functional.resample(
                    torch.from_numpy(waveform_float).unsqueeze(0),
                    orig_freq=model.sampling_rate,
                    new_freq=output_sampling_rate,
                )
                .squeeze(0)
                .numpy()
            )
        waveform = waveform_float.clip(-1.0, 1.0).astype(np.float32)
        last_audio_path = "last_audio"
        os.makedirs(last_audio_path, exist_ok=True)
        if ref_audio:
            ref_basename = os.path.basename(ref_audio).rpartition(".")[0]
        elif instruct and instruct.strip():
            ref_basename = "voice_design"
        else:
            ref_basename = "auto"
        speed_label = speed if speed is not None else 1.0
        filename = (
            f"{_safe_filename_part(model_name)}--"
            f"{_safe_filename_part(ref_basename)}--"
            f"spd{speed_label}.wav"
        )
        output_path = os.path.join(last_audio_path, filename)
        sf.write(output_path, waveform, output_sampling_rate, subtype="PCM_32")
        _elapsed = time.time() - _t0
        logging.info(
            "[推理完成] 模型=%s 耗时=%.2fs 输出=%s",
            model_name, _elapsed, output_path,
        )
        return output_path, "完成。", output_path

    def _save_edited_audio(audio_path, target_path):
        if not audio_path:
            return None, "没有可保存的音频。", target_path, None

        if not target_path:
            target_path = os.path.join("last_audio", "edited_audio.wav")

        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)

        try:
            data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
            if sample_rate != 48000:
                audio_tensor = torch.from_numpy(np.asarray(data, dtype=np.float32))
                if audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                else:
                    audio_tensor = audio_tensor.T
                data = (
                    torchaudio.functional.resample(
                        audio_tensor,
                        orig_freq=sample_rate,
                        new_freq=48000,
                    )
                    .T.squeeze()
                    .numpy()
                )
                sample_rate = 48000
            sf.write(target_path, data, sample_rate, subtype="PCM_32")
        except Exception as e:
            return audio_path, f"保存失败：{type(e).__name__}: {e}", target_path, None

        return (
            target_path,
            f"已保存：{os.path.basename(target_path)}",
            target_path,
            target_path,
        )

    def _transcribe_one(audio_path: str, add_punctuation: bool) -> str:
        """转录单个音频，返回文本字符串（失败时返回空串）"""
        MAX_REF_AUDIO_DURATION = 15
        if not audio_path or not os.path.exists(audio_path):
            return ""
        try:
            info = sf.info(audio_path)
            if info.duration > MAX_REF_AUDIO_DURATION:
                gr.Warning(f"{os.path.basename(audio_path)} 过长（{info.duration:.1f}s > {MAX_REF_AUDIO_DURATION}s），已跳过转录")
                return ""
        except Exception:
            pass
        try:
            return asr_sherpaonnx.transcribe(audio_path, add_punctuation=bool(add_punctuation))
        except Exception as e:
            gr.Warning(f"{os.path.basename(audio_path)} 转录失败：{e}")
            return ""

    def _transcribe_ref_audios(audio_files, add_punctuation):
        """批量转录：支持单个或多个参考音频文件。
        audio_files: None | str | list[str] — gr.File 返回的路径或路径列表
        """
        if not audio_files:
            return gr.update(), "请上传参考音频。"

        # 统一为列表
        if isinstance(audio_files, str):
            paths = [audio_files]
        elif isinstance(audio_files, dict):          # gr.File 单文件有时返回 dict
            paths = [audio_files.get("name", "") or audio_files.get("path", "")]
        else:
            paths = [
                (f.get("name") or f.get("path") if isinstance(f, dict) else f)
                for f in audio_files
            ]
        paths = [p for p in paths if p]

        if not paths:
            return gr.update(), "请上传参考音频。"

        results = [_transcribe_one(p, add_punctuation) for p in paths]

        if len(results) == 1:
            status = "参考音频已转录，可直接修改参考文本。"
        else:
            status = f"已批量转录 {len(results)} 个参考音频，每行对应一个音频的文本。"

        combined = "\n".join(results)
        return gr.update(value=combined), status

    # Allow external wrappers (e.g. spaces.GPU for ZeroGPU Spaces)
    _gen = generate_fn if generate_fn is not None else _gen_core

    def _batch_gen_fn(
        model_name, text, language,
        audio_files,
        ref_text_block,
        final_instruct,
        ns, gs, dn, sp, du, pp, po,
    ):
        """批量生成：每行 text 对应一个参考音频文件，检查数量一致后逐条生成，拼接为单个音频文件输出。"""
        # --- 解析文件列表 ---
        if isinstance(audio_files, str):
            paths = [audio_files]
        elif isinstance(audio_files, dict):
            paths = [audio_files.get("name", "") or audio_files.get("path", "")]
        elif audio_files:
            paths = [
                (f.get("name") or f.get("path") if isinstance(f, dict) else f)
                for f in audio_files
            ]
        else:
            paths = []
        paths = [p for p in paths if p]

        # --- 解析文本行 ---
        lines = [l for l in (text or "").splitlines() if l.strip()]

        if not lines:
            return None, "请输入要合成的文本（每行一条）。", None, None

        n_audio = len(paths)
        n_lines = len(lines)

        if n_audio > 1 and n_lines != n_audio:
            msg = (
                f"⚠️ 数量不一致：生成文本有 {n_lines} 行，"
                f"但上传了 {n_audio} 个参考音频。\n"
                f"请确保每行文本对应一个参考音频（共 {n_audio} 行）。"
            )
            gr.Warning(msg)
            return None, msg, None, None

        # --- 解析参考文本行（可为空） ---
        ref_lines = [l for l in (ref_text_block or "").splitlines() if l.strip()]
        # 如果参考文本行数与音频数一致则逐条对应，否则全部用第一条或空
        def _ref_text_for(idx):
            if len(ref_lines) > idx:
                return ref_lines[idx]
            elif len(ref_lines) == 1:
                return ref_lines[0]
            return None

        output_sampling_rate = 48000
        gen_dir = "gen_audio"
        os.makedirs(gen_dir, exist_ok=True)
        delete_old_files_and_dirs(gen_dir, days=2)

        generated_paths: List[str] = []
        errors: List[str] = []

        for i, (line_text, ref_path) in enumerate(zip(lines, paths if paths else [None] * n_lines)):
            ref_basename = os.path.basename(ref_path).rpartition(".")[0] if ref_path else "auto"
            fname = f"{i+1:03d}__{_safe_filename_part(ref_basename)}.wav"
            out_path = os.path.join(gen_dir, fname)

            _, status_msg, saved_path = _gen(
                model_name,
                line_text,
                language,
                ref_path,
                final_instruct,
                ns, gs, dn, sp, du, pp, po,
                ref_text=_ref_text_for(i),
            )
            if saved_path and os.path.exists(saved_path):
                shutil.copy2(saved_path, out_path)
                generated_paths.append(out_path)
            else:
                errors.append(f"第 {i+1} 条失败：{status_msg}")

        if not generated_paths:
            return None, "批量生成全部失败：\n" + "\n".join(errors), None, None

        status = f"批量生成完成，共 {len(generated_paths)} 条"
        if errors:
            status += f"，{len(errors)} 条失败：" + "；".join(errors)
        status += "。"

        # --- 拼接所有生成音频为单文件 ---
        segments = []
        for p in generated_paths:
            data, sr = sf.read(p, dtype="float32", always_2d=False)
            if sr != output_sampling_rate:
                t = torch.from_numpy(np.asarray(data, dtype=np.float32))
                if t.ndim == 1:
                    t = t.unsqueeze(0)
                else:
                    t = t.T
                data = (
                    torchaudio.functional.resample(t, orig_freq=sr, new_freq=output_sampling_rate)
                    .T.squeeze()
                    .numpy()
                )
            # 统一为 (1, T) 供 cross_fade_chunks 使用
            if data.ndim == 1:
                data = data[np.newaxis, :]   # (T,) → (1, T)
            elif data.ndim == 2:
                data = data.T                # (T, C) → (C, T)
            segments.append(data)
        # 拼接：加 0.5 s 静音区 + 交叉淡化
        merged_audio = cross_fade_chunks(
            segments,
            sample_rate=output_sampling_rate,
            silence_duration=1.0,
        ).squeeze(0)  # (1, T) → (T,)

        # 文件名：与 _gen_core 保持一致，ref_basename 为各参考音频名以 "+" 拼接
        # 文件系统限制：Linux/macOS 255 字节，Windows NTFS 255 字符；统一按字节控制
        _FNAME_MAX_BYTES = 255
        ref_stems = [
            _safe_filename_part(os.path.basename(p).rpartition(".")[0])
            for p in (paths if paths else [])
        ]
        speed_label = sp if sp is not None else 1.0
        ref_basename = "+".join(ref_stems) if ref_stems else "batch"
        # 先算出除 ref_basename 外的固定字节开销
        fixed = (
            f"{_safe_filename_part(model_name)}----spd{speed_label}.wav"
        ).encode("utf-8")
        max_ref_bytes = _FNAME_MAX_BYTES - len(fixed)
        if len(ref_basename.encode("utf-8")) > max_ref_bytes:
            suffix = f"+…({len(ref_stems)})"
            budget = max_ref_bytes - len(suffix.encode("utf-8"))
            # 按字节截断，再安全解码（避免切断多字节字符）
            ref_basename = (
                ref_basename.encode("utf-8")[:budget].decode("utf-8", errors="ignore")
                + suffix
            )
        merged_fname = (
            f"{_safe_filename_part(model_name)}--"
            f"{_safe_filename_part(ref_basename)}--"
            f"spd{speed_label}.wav"
        )
        merged_path = os.path.join(gen_dir, merged_fname)
        sf.write(merged_path, merged_audio, output_sampling_rate, subtype="PCM_32")

        return merged_path, status, merged_path, merged_path

    # =====================================================================
    # UI
    # =====================================================================
    theme = gr.themes.Soft(
        font=["Inter", "Arial", "sans-serif"],
    )
    css = """
    .gradio-container {max-width: 100% !important; font-size: 16px !important;}
    .gradio-container h1 {font-size: 1.5em !important;}
    .gradio-container .prose {font-size: 1.1em !important;}
    .compact-audio audio {height: 60px !important;}
    .compact-audio .waveform {min-height: 80px !important;}
    #vc_download_file {display: none !important;}
    """

    auto_download_js = """
    () => {
        setTimeout(() => {
            const links = document.querySelectorAll('#vc_download_file a[href]');
            const link = links[links.length - 1];
            if (link) link.click();
        }, 500);
        return [];
    }
    """

    def _lang_dropdown(label="语言（可选）", value=_AUTO_LABEL):
        return gr.Dropdown(
            label=label,
            choices=_ALL_LANGUAGES,
            value=value,
            allow_custom_value=False,
            interactive=True,
        )

    with gr.Blocks(theme=theme,analytics_enabled=False, css=css) as demo:
        with gr.Row():
            target_lang = _lang_dropdown()
            model_select = gr.Dropdown(
                label="模型",
                choices=list(model_choices.keys()),
                value=default_model_name,
                interactive=True,
            )
        with gr.Row(equal_height=True):
            target_text = gr.Textbox(
                label="生成文本",
                lines=4,
                placeholder="请输入要合成的文本……（批量模式：每行一条，行数需与参考音频个数一致）",
            )
            out_audio = gr.Audio(
                    label="合成结果",
                    type="filepath",
                    autoplay=True,
                    interactive=True,
                    show_download_button=False,
                    show_share_button=False,
                    sources=[],
                )
            out_audio_path = gr.State(value=None)
            download_file = gr.File(
                    label="下载文件",
                    elem_id="vc_download_file",
                    show_label=False,
                    file_count="multiple",
                    visible=True,
                )
            out_status = gr.Textbox(label="状态", lines=2)
        with gr.Row():
            btn_gen = gr.Button("🚀 立即生成", variant="primary")
            btn_save = gr.Button("💾 下载")
        # 合并后的参考音频路径（单/多文件共享）
        merged_ref_audio = gr.State(value=None)

        with gr.Row(equal_height=True):
            with gr.Column():
                with gr.Tabs() as ref_audio_tabs:
                    with gr.Tab("🎙️ 单参考音频"):
                        ref_audio_single = gr.Audio(
                            label="参考音频（可选，支持录音，提供时启用克隆）",
                            type="filepath",
                            sources=["upload", "microphone"],
                            elem_classes="compact-audio",
                        )
                    with gr.Tab("📂 多参考批量"):
                        ref_audio_multi = gr.File(
                            label="参考音频（支持多文件批量克隆）",
                            file_count="multiple",
                            file_types=[".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"],
                            elem_classes="compact-audio",
                        )
            ref_text = gr.Textbox(
                    label="参考音频文本（可选；批量时每行对应一个音频）",
                    lines=10,
                    placeholder="参考音频对应文本。批量上传时每行对应一个音频的转录文本。",
                )
            with gr.Column():
                set_sp = gr.Slider(
                    0.5, 1.5, value=1.0, step=0.05, label="语速", info="1.0 为正常语速，大于 1 更快，小于 1 更慢。"
                )
                ref_punctuation = gr.Checkbox(
                    label="自动参考文本包含标点",
                    value=False,
                )
                        
        # 高级设置与设计选项
        with gr.Row():
            with gr.Accordion("🎨 声音引导提示词 (可选)", open=True):
                _AUTO = _AUTO_LABEL
                vd_groups = []
                cats = list(_CATEGORIES.items())
                with gr.Row():
                    with gr.Column():
                        for _cat, _choices in cats[:2]:
                            vd_groups.append(
                                gr.Dropdown(label=_cat, choices=[_AUTO] + _choices, value=_AUTO)
                            )
                    with gr.Column():
                        for _cat, _choices in cats[2:4]:
                            vd_groups.append(
                                gr.Dropdown(label=_cat, choices=[_AUTO] + _choices, value=_AUTO)
                            )
                    with gr.Column():
                        for _cat, _choices in cats[4:]:
                            vd_groups.append(
                                gr.Dropdown(label=_cat, choices=[_AUTO] + _choices, value=_AUTO)
                            )
                instruct_text = gr.Textbox(label="生成的提示词 (Instruct)", lines=1, interactive=False)

                def _update_instruct(*groups):
                    selected = [g for g in groups if g and g != _AUTO_LABEL]
                    parts = []
                    for v in selected:
                        if " / " in v:
                            en, zh = v.split(" / ", 1)
                            if "Dialect" in v.split(" / ")[0]:
                                parts.append(zh.strip())
                            else:
                                parts.append(en.strip())
                        else:
                            parts.append(v)
                    return ", ".join(parts)

                for dd in vd_groups:
                    dd.change(_update_instruct, inputs=vd_groups, outputs=instruct_text)

                _auto_val = _AUTO_LABEL
                _n_vd = len(vd_groups)

                def _reset_vd_groups(_):
                    return tuple(gr.update(value=_auto_val) for _ in range(_n_vd))

                model_select.change(
                    _reset_vd_groups,
                    inputs=[model_select],
                    outputs=list(vd_groups),
                    queue=False,
                )

            with gr.Accordion("⚙️ 高级生成设置", open=True):
                    
                with gr.Row():
                    set_du = gr.Number(value=None, label="固定时长（秒）", info="留空则使用语速控制。")
                    set_ns = gr.Slider(4, 64, value=32, step=1, label="推理步数", info="默认 32。")
                    set_gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="引导强度（CFG）", info="默认 2.0。")
                with gr.Row():
                    set_dn = gr.Checkbox(label="降噪", value=True, info="默认关闭。")
                    set_pp = gr.Checkbox(label="预处理参考音频", value=True, info="静音移除、裁剪、补充标点。")
                    set_po = gr.Checkbox(label="后处理输出音频", value=True, info="移除长静音。")


        def _get_paths(audio_files):
            """统一解析 gr.Audio / gr.File 返回值为路径列表。"""
            if not audio_files:
                return []
            if isinstance(audio_files, str):
                return [audio_files]
            if isinstance(audio_files, dict):
                p = audio_files.get("name") or audio_files.get("path", "")
                return [p] if p else []
            result = []
            for f in audio_files:
                if isinstance(f, dict):
                    p = f.get("name") or f.get("path", "")
                    if p:
                        result.append(p)
                elif isinstance(f, str) and f:
                    result.append(f)
            return result

        def _sync_single(path):
            """gr.Audio 单文件上传/录音 → 更新 merged_ref_audio State。"""
            return [path] if path else None

        def _sync_multi(files):
            """gr.File 多文件上传 → 更新 merged_ref_audio State。"""
            paths = _get_paths(files)
            return paths if paths else None

        def _unified_fn(
            model_name, text, lang,
            r_aud, r_txt,
            final_instruct,
            ns, gs, dn, sp, du, pp, po
        ):
            try:
                if not final_instruct or not final_instruct.strip():
                    final_instruct = None
                else:
                    final_instruct = final_instruct.strip()

                paths = _get_paths(r_aud)
                n_audio = len(paths)

                # 批量模式：多个音频文件
                if n_audio > 1:
                    audio_out, status, preview, file_paths = _batch_gen_fn(
                        model_name, text, lang,
                        paths,
                        r_txt,
                        final_instruct,
                        ns, gs, dn, sp, du, pp, po,
                    )
                    return (
                        audio_out,
                        status,
                        preview,
                        file_paths,
                    )
                else:
                    # 单个模式（0 或 1 个参考音频）
                    single_path = paths[0] if paths else None
                    audio_out, status, saved = _gen(
                        model_name,
                        text,
                        lang,
                        single_path,
                        final_instruct,
                        ns, gs, dn, sp, du, pp, po,
                        ref_text=r_txt or None,
                    )
                    return audio_out, status, saved, saved
            except gr.Error:
                raise  # _gen_core 已记录日志，直接透传
            except Exception as e:
                logging.error(
                    "[顶层异常] 模型=%s 文本(前200字)=%s 异常: %s",
                    model_name,
                    (text or "").strip()[:200],
                    e,
                    exc_info=True,
                )
                raise gr.Error(f"{type(e).__name__}: {e}")

        # --- 同步 State 的回调 ---
        ref_audio_single.change(
            _sync_single,
            inputs=[ref_audio_single],
            outputs=[merged_ref_audio],
            queue=False,
        )
        ref_audio_multi.upload(
            _sync_multi,
            inputs=[ref_audio_multi],
            outputs=[merged_ref_audio],
            queue=False,
        )
        ref_audio_multi.clear(
            lambda: (None, gr.update(value="")),
            outputs=[merged_ref_audio, ref_text],
            queue=False,
        )

        btn_gen.click(
            _unified_fn,
            inputs=[
                model_select,
                target_text,
                target_lang,
                merged_ref_audio,
                ref_text,
                instruct_text,
                set_ns,
                set_gs,
                set_dn,
                set_sp,
                set_du,
                set_pp,
                set_po,
            ],
            outputs=[out_audio, out_status, out_audio_path, download_file],
            concurrency_id="gpu_infer",
            concurrency_limit=concurrency_limit,
        )
        # 单文件上传/录音 → 自动转录（清除时跳过）
        ref_audio_single.change(
            lambda path, punct: (
                _transcribe_ref_audios([path], punct) if path
                else (gr.update(value=""), gr.update())
            ),
            inputs=[ref_audio_single, ref_punctuation],
            outputs=[ref_text, out_status],
            concurrency_id="asr",
            concurrency_limit=4,
        )
        # 多文件上传 → 自动转录
        ref_audio_multi.upload(
            _transcribe_ref_audios,
            inputs=[ref_audio_multi, ref_punctuation],
            outputs=[ref_text, out_status],
            concurrency_id="asr",
            concurrency_limit=4,
        )
        ref_punctuation.change(
            fn=None,
            inputs=[ref_punctuation],
            js="""
            (value) => {
                try {
                    localStorage.setItem(
                        "omnivoice_ref_text_add_punctuation",
                        value ? "true" : "false"
                    );
                } catch (e) {}
                return [];
            }
            """,
            queue=False,
        )
        set_dn.change(
            fn=None,
            inputs=[set_dn],
            js="""
            (value) => {
                try {
                    localStorage.setItem(
                        "omnivoice_set_dn",
                        value ? "true" : "false"
                    );
                } catch (e) {}
                return [];
            }
            """,
            queue=False,
        )
        demo.load(
            fn=None,
            outputs=[ref_punctuation, set_dn],
            js="""
            () => {
                try {
                    return [
                        localStorage.getItem(
                            "omnivoice_ref_text_add_punctuation"
                        ) === "true",
                        localStorage.getItem(
                            "omnivoice_set_dn"
                        ) === "true"
                    ];
                } catch (e) {
                    return [false, false];
                }
            }
            """,
            queue=False,
        )
        save_event = btn_save.click(
            _save_edited_audio,
            inputs=[out_audio, out_audio_path],
            outputs=[out_audio, out_status, out_audio_path, download_file],
        )
        save_event.then(fn=None, js=auto_download_js, queue=False)

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    log_file = os.path.abspath("omnivoice_demo.log")
    _file_handler = logging.FileHandler(log_file, encoding="utf-8")
    _file_handler.setLevel(logging.ERROR)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            _file_handler,
        ],
    )
    logging.info("日志文件路径: %s", log_file)

    # 兜底：捕获后台线程中未处理的异常并写入日志
    def _thread_excepthook(args):
        if args.exc_type is SystemExit:
            return
        logging.error(
            "[线程未捕获异常] 线程=%s",
            args.thread,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
    threading.excepthook = _thread_excepthook
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    fallback_model = args.model
    if not fallback_model:
        parser.print_help()
        return 0
    model_choices = discover_models(args.model_root, fallback_model)
    logging.info("Discovered models: %s", model_choices)

    def _load_model(model_id: str) -> OmniVoice:
        logging.info("Loading model from %s, device=%s ...", model_id, device)
        if model_id == "默认":
            model_id = "k2-fsa/OmniVoice"
        model = OmniVoice.from_pretrained(
            model_id,
            device_map=device,
            dtype=torch.float16
        )
        model.eval()
        return model

    concurrency_limit = max(1, int(args.concurrency))
    model_cache = ModelCache(
        device=device,
        gpu_slots=concurrency_limit,
        load_fn=_load_model,
    )

    logging.info("Gradio GPU inference concurrency limit: %d", concurrency_limit)
    logging.info("Model LRU GPU slots: %d", concurrency_limit)
    demo = build_demo(
        model_cache,
        model_choices,
        concurrency_limit=concurrency_limit,
    )

    import tempfile
    allowed_paths = [
        os.path.abspath("gradio_tmp"),
        os.path.abspath("gen_audio"),
        os.path.abspath("last_audio"),
    ]
    try:
        allowed_paths.append(tempfile.gettempdir())
    except Exception:
        pass

    queued_demo = demo.queue(default_concurrency_limit=concurrency_limit)

    # FastAPI 全局异常 handler：捕获序列化层以下的崩溃，避免前端收到无法解析的响应
    try:
        from fastapi import Request
        from fastapi.responses import JSONResponse

        @queued_demo.app.exception_handler(Exception)
        async def _global_exception_handler(request: Request, exc: Exception):
            logging.error(
                "[FastAPI全局异常] %s %s 异常: %s",
                request.method,
                request.url,
                exc,
                exc_info=True,
            )
            return JSONResponse(
                status_code=500,
                content={"detail": f"{type(exc).__name__}: {exc}"},
            )
    except Exception:
        logging.warning("无法注册 FastAPI 全局异常 handler，忽略。", exc_info=True)

    queued_demo.launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
        allowed_paths=allowed_paths,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
