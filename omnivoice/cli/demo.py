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

import argparse
import collections
import gc
import logging
import os
import threading
from typing import Any, Callable, Dict

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name


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
        default="k2-fsa/OmniVoice",
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
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0).")
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
        help="Skip loading sherpa-onnx ASR model. Reference text auto-transcription"
        " will be unavailable.",
    )
    parser.add_argument(
        "--asr-model",
        default="csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14",
        help="sherpa-onnx paraformer ONNX path, model directory, or HuggingFace repo id"
        " (default: csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14).",
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
        mode,
        ref_text=None,
    ):
        model_id = model_choices.get(model_name)
        if model_id is None:
            return None, f"未知模型：{model_name}"
        if not text or not text.strip():
            return None, "请输入要合成的文本。"

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

        if mode == "clone" and not ref_audio:
            return None, "请上传参考音频。"

        acquired = False
        try:
            with infer_semaphore, torch.inference_mode():
                model = model_cache.acquire(model_id)
                acquired = True
                if mode == "clone":
                    kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                    )

                if instruct and instruct.strip():
                    kw["instruct"] = instruct.strip()

                audio = model.generate(**kw)
        except Exception as e:
            return None, f"错误：{type(e).__name__}: {e}"
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
        if mode == "clone" and ref_audio:
            ref_basename = os.path.basename(ref_audio).rpartition(".")[0]
        else:
            ref_basename = "voice_design"
        speed_label = speed if speed is not None else 1.0
        filename = (
            f"{_safe_filename_part(model_name)}--"
            f"{_safe_filename_part(ref_basename)}--"
            f"spd{speed_label}-orgi_audio.wav"
        )
        output_path = os.path.join(last_audio_path, filename)
        sf.write(output_path, waveform, output_sampling_rate, subtype="PCM_32")
        return output_path, "完成。"

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
    """

    def _lang_dropdown(label="语言（可选）", value=_AUTO_LABEL):
        return gr.Dropdown(
            label=label,
            choices=_ALL_LANGUAGES,
            value=value,
            allow_custom_value=False,
            interactive=True,
            info="保持为自动时由模型自行判断语言。",
        )

    def _gen_settings():
        with gr.Accordion("生成设置（可选）", open=False):
            sp = gr.Slider(
                0.5,
                1.5,
                value=1.0,
                step=0.05,
                label="语速",
                info="1.0 为正常语速，大于 1 更快，小于 1 更慢。设置固定时长后将忽略此项。",
            )
            du = gr.Number(
                value=None,
                label="固定时长（秒）",
                info="留空则使用语速控制。填写固定时长后会覆盖语速设置。",
            )
            ns = gr.Slider(
                4,
                64,
                value=32,
                step=1,
                label="推理步数",
                info="默认 32。数值越低速度越快，数值越高质量通常更好。",
            )
            dn = gr.Checkbox(
                label="降噪",
                value=True,
                info="默认开启。取消勾选可关闭降噪。",
            )
            gs = gr.Slider(
                0.0,
                4.0,
                value=2.0,
                step=0.1,
                label="引导强度（CFG）",
                info="默认 2.0。",
            )
            pp = gr.Checkbox(
                label="预处理参考音频",
                value=True,
                info="对参考音频进行静音移除和裁剪，并在参考文本末尾补充标点。",
            )
            po = gr.Checkbox(
                label="后处理输出音频",
                value=True,
                info="移除生成音频中的长静音。",
            )
        return ns, gs, dn, sp, du, pp, po

    with gr.Blocks(theme=theme, css=css, title="OmniVoice 演示") as demo:
        gr.Markdown(
            """
# OmniVoice 演示

支持声音克隆和声音设计，可用于多语言文本转语音生成。
"""
        )
        model_select = gr.Dropdown(
            label="模型",
            choices=list(model_choices.keys()),
            value=default_model_name,
            interactive=True,
        )

        with gr.Tabs():
            # ==============================================================
            # Voice Clone
            # ==============================================================
            with gr.TabItem("声音克隆"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(
                            label="待合成文本",
                            lines=4,
                            placeholder="请输入要合成的文本...",
                        )
                        vc_ref_audio = gr.Audio(
                            label="参考音频",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        gr.Markdown(
                            "<span style='font-size:0.85em;color:#888;'>"
                            "建议上传 3 到 10 秒的参考音频。"
                            "</span>"
                        )
                        vc_ref_text = gr.Textbox(
                            label="参考音频文本（可选）",
                            lines=2,
                            placeholder="参考音频对应文本。留空时将使用 ASR 自动识别。",
                        )
                        vc_lang = _lang_dropdown("语言（可选）")
                        with gr.Accordion("提示词（可选）", open=False):
                            vc_instruct = gr.Textbox(label="提示词", lines=2)
                        (
                            vc_ns,
                            vc_gs,
                            vc_dn,
                            vc_sp,
                            vc_du,
                            vc_pp,
                            vc_po,
                        ) = _gen_settings()
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(
                            label="合成结果",
                            type="filepath",
                            autoplay=True,
                            interactive=True,
                            sources=[],
                        )
                        vc_btn = gr.Button("生成", variant="primary")
                        vc_status = gr.Textbox(label="状态", lines=2)

                def _clone_fn(
                    model_name,
                    text,
                    lang,
                    ref_aud,
                    ref_text,
                    instruct,
                    ns,
                    gs,
                    dn,
                    sp,
                    du,
                    pp,
                    po,
                ):
                    return _gen(
                        model_name,
                        text,
                        lang,
                        ref_aud,
                        instruct,
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="clone",
                        ref_text=ref_text or None,
                    )

                vc_btn.click(
                    _clone_fn,
                    inputs=[
                        model_select,
                        vc_text,
                        vc_lang,
                        vc_ref_audio,
                        vc_ref_text,
                        vc_instruct,
                        vc_ns,
                        vc_gs,
                        vc_dn,
                        vc_sp,
                        vc_du,
                        vc_pp,
                        vc_po,
                    ],
                    outputs=[vc_audio, vc_status],
                    concurrency_id="gpu_infer",
                    concurrency_limit=concurrency_limit,
                )

            # ==============================================================
            # Voice Design
            # ==============================================================
            with gr.TabItem("声音设计"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(
                            label="待合成文本",
                            lines=4,
                            placeholder="请输入要合成的文本...",
                        )
                        vd_lang = _lang_dropdown()

                        _AUTO = _AUTO_LABEL
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            vd_groups.append(
                                gr.Dropdown(
                                    label=_cat,
                                    choices=[_AUTO] + _choices,
                                    value=_AUTO,
                                    info=_ATTR_INFO.get(_cat),
                                )
                            )

                        (
                            vd_ns,
                            vd_gs,
                            vd_dn,
                            vd_sp,
                            vd_du,
                            vd_pp,
                            vd_po,
                        ) = _gen_settings()
                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(
                            label="合成结果",
                            type="filepath",
                            autoplay=True,
                            interactive=True,
                            sources=[],
                        )
                        vd_btn = gr.Button("生成", variant="primary")
                        vd_status = gr.Textbox(label="状态", lines=2)

                def _build_instruct(groups):
                    """Extract instruct text from UI dropdowns.

                    Language unification and validation is handled by
                    _resolve_instruct inside _preprocess_all.
                    """
                    selected = [g for g in groups if g and g != _AUTO_LABEL]
                    if not selected:
                        return None
                    parts = []
                    for v in selected:
                        if " / " in v:
                            en, zh = v.split(" / ", 1)
                            # Dialects have no English equivalent
                            if "Dialect" in v.split(" / ")[0]:
                                parts.append(zh.strip())
                            else:
                                parts.append(en.strip())
                        else:
                            parts.append(v)
                    return ", ".join(parts)

                def _design_fn(
                    model_name, text, lang, ns, gs, dn, sp, du, pp, po, *groups
                ):
                    return _gen(
                        model_name,
                        text,
                        lang,
                        None,
                        _build_instruct(groups),
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="design",
                    )

                vd_btn.click(
                    _design_fn,
                    inputs=[
                        model_select,
                        vd_text,
                        vd_lang,
                        vd_ns,
                        vd_gs,
                        vd_dn,
                        vd_sp,
                        vd_du,
                        vd_pp,
                        vd_po,
                    ]
                    + vd_groups,
                    outputs=[vd_audio, vd_status],
                    concurrency_id="gpu_infer",
                    concurrency_limit=concurrency_limit,
                )

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
        model = OmniVoice.from_pretrained(
            model_id,
            device_map=device,
            dtype=torch.float16,
            load_asr=not args.no_asr,
            asr_model_name=args.asr_model,
            asr_num_threads=args.asr_threads,
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

    demo.queue(default_concurrency_limit=concurrency_limit).launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
