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
from datetime import time
from aiofiles import tempfile
import argparse
import collections
import gc
import logging
import os
import threading
from typing import Any, Callable, Dict

# Set Gradio temp directory to a local folder to avoid tmpfs RAM consumption on Linux
if "GRADIO_TEMP_DIR" not in os.environ:
    os.environ["GRADIO_TEMP_DIR"] = os.path.abspath("gradio_tmp")
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
from omnivoice.utils import asr_sherpaonnx

# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_AUTO_LABEL = "自动"
_ALL_LANGUAGES = [_AUTO_LABEL] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# Voice Design instruction templates
# ---------------------------------------------------------------------------
# Each option is displayed as "English / 中文".
# The model expects English for accents and Chinese for dialects.
_CATEGORIES = {
    "Gender / 性别": ["Male / 男", "Female / 女"],
    "Age / 年龄": [
        "Child / 儿童",
        "Teenager / 少年",
        "Young Adult / 青年",
        "Middle-aged / 中年",
        "Elderly / 老年",
    ],
    "Pitch / 音调": [
        "Very Low Pitch / 极低音调",
        "Low Pitch / 低音调",
        "Moderate Pitch / 中音调",
        "High Pitch / 高音调",
        "Very High Pitch / 极高音调",
    ],
    "Style / 风格": ["Whisper / 耳语"],
    "English Accent / 英文口音": [
        "American Accent / 美式口音",
        "Australian Accent / 澳大利亚口音",
        "British Accent / 英国口音",
        "Chinese Accent / 中国口音",
        "Canadian Accent / 加拿大口音",
        "Indian Accent / 印度口音",
        "Korean Accent / 韩国口音",
        "Portuguese Accent / 葡萄牙口音",
        "Russian Accent / 俄罗斯口音",
        "Japanese Accent / 日本口音",
    ],
    "Chinese Dialect / 中文方言": [
        "Henan Dialect / 河南话",
        "Shaanxi Dialect / 陕西话",
        "Sichuan Dialect / 四川话",
        "Guizhou Dialect / 贵州话",
        "Yunnan Dialect / 云南话",
        "Guilin Dialect / 桂林话",
        "Jinan Dialect / 济南话",
        "Shijiazhuang Dialect / 石家庄话",
        "Gansu Dialect / 甘肃话",
        "Ningxia Dialect / 宁夏话",
        "Qingdao Dialect / 青岛话",
        "Northeast Dialect / 东北话",
    ],
}

_ATTR_INFO = {
    "English Accent / 英文口音": "Only effective for English speech.",
    "Chinese Dialect / 中文方言": "Only effective for Chinese speech.",
}

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
            return None, f"未知模型：{model_name}", None
        if not text or not text.strip():
            return None, "请输入要合成的文本。", None

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
            return None, f"错误：{type(e).__name__}: {e}", None
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
            f"spd{speed_label}--last_audio.wav"
        )
        output_path = os.path.join(last_audio_path, filename)
        sf.write(output_path, waveform, output_sampling_rate, subtype="PCM_32")
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

    def _transcribe_ref_audio(audio_path, add_punctuation):
        """转录前先检测音频时长，超过限制直接返回提示，避免长音频占用转录队列"""
        MAX_REF_AUDIO_DURATION = 15
        if not audio_path or not os.path.exists(audio_path):
            return gr.update(), "请上传参考音频。"
        try:
            info = sf.info(audio_path)
            if info.duration > MAX_REF_AUDIO_DURATION:
                gr.Warning(f"参考音频过长（{info.duration:.1f}s），请上传 {MAX_REF_AUDIO_DURATION}s 以内的音频")
                return gr.update(value=""), "你输入的音频过长"
        except Exception as e:
            return gr.update(), f"读取音频时长失败: {e}"
            
        try:
            text = asr_sherpaonnx.transcribe(
                audio_path, add_punctuation=bool(add_punctuation)
            )
            return gr.update(value=text), "参考音频已转录，可直接修改参考文本。"
        except Exception as e:
            return gr.update(), f"转录失败：{e}"

    # Allow external wrappers (e.g. spaces.GPU for ZeroGPU Spaces)
    _gen = generate_fn if generate_fn is not None else _gen_core

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
            const links = document.querySelectorAll('#vc_download_file a[href], #vd_download_file a[href]');
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
                placeholder="请输入要合成的文本...",
            )
            out_audio = gr.Audio(
                    label="合成结果",
                    type="filepath",
                    autoplay=True,
                    interactive=True,
                    sources=[],
                )
            out_audio_path = gr.State(value=None)
            download_file = gr.File(
                    label="下载文件",
                    elem_id="vc_download_file",
                    show_label=False,
                    visible=True,
                )
            out_status = gr.Textbox(label="状态", lines=2)
        with gr.Row():
            btn_gen = gr.Button("🚀 立即生成", variant="primary")
            btn_save = gr.Button("💾 下载")
        with gr.Row(equal_height=True):
                ref_audio = gr.Audio(
                    label="参考音频（可选，提供时启用克隆）",
                    type="filepath",
                    elem_classes="compact-audio",
                )
                ref_text = gr.Textbox(
                        label="参考音频文本（可选）",
                        lines=10,
                        placeholder="参考音频对应文本。",
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

            with gr.Accordion("⚙️ 高级生成设置", open=True):
                    
                with gr.Row():
                    set_du = gr.Number(value=None, label="固定时长（秒）", info="留空则使用语速控制。")
                    set_ns = gr.Slider(4, 64, value=32, step=1, label="推理步数", info="默认 32。")
                    set_gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="引导强度（CFG）", info="默认 2.0。")
                with gr.Row():
                    set_dn = gr.Checkbox(label="降噪", value=False, info="默认关闭。")
                    set_pp = gr.Checkbox(label="预处理参考音频", value=True, info="静音移除、裁剪、补充标点。")
                    set_po = gr.Checkbox(label="后处理输出音频", value=True, info="移除长静音。")


        def _unified_fn(
            model_name, text, lang,
            r_aud, r_txt,
            final_instruct,
            ns, gs, dn, sp, du, pp, po
        ):
            if not final_instruct or not final_instruct.strip():
                final_instruct = None
            else:
                final_instruct = final_instruct.strip()
            
            return _gen(
                model_name,
                text,
                lang,
                r_aud,
                final_instruct,
                ns,
                gs,
                dn,
                sp,
                du,
                pp,
                po,
                ref_text=r_txt or None,
            )

        btn_gen.click(
            _unified_fn,
            inputs=[
                model_select,
                target_text,
                target_lang,
                ref_audio,
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
            outputs=[out_audio, out_status, out_audio_path],
            concurrency_id="gpu_infer",
            concurrency_limit=concurrency_limit,
        )
        ref_audio.upload(
            _transcribe_ref_audio,
            inputs=[ref_audio, ref_punctuation],
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
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
    allowed_paths = [os.path.abspath("gradio_tmp")]
    try:
        allowed_paths.append(tempfile.gettempdir())
    except Exception:
        pass

    demo.queue(default_concurrency_limit=concurrency_limit).launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
        allowed_paths=allowed_paths,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
