# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# 1. 收集依赖包的子模块，防止 pyinstaller 静态分析遗漏动态导入的库
hiddenimports = (
    collect_submodules('omnivoice') +
    collect_submodules('gradio') +
    collect_submodules('sherpa_onnx') +
    collect_submodules('soundfile') +
    [
        'numpy',
        'torch',
        'torchaudio',
        'transformers',
        'librosa',
        'pydub',
        'accelerate',
        'huggingface_hub',
        'tqdm'
    ]
)

# 2. 收集静态文件和非 python 资源文件
datas = []
# 必须收集 gradio 的前端静态网页资源 (templates/themes/etc.)
datas += collect_data_files('gradio', include_py_files=True)
# 收集 transformers 依赖的相关配置与文件
datas += collect_data_files('transformers')
# 将本项目的 omnivoice 源码作为数据文件复制，以保证某些相对路径的读取正常
datas += [('omnivoice', 'omnivoice')]

binaries = []

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='omnivoice-demo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # 开启控制台，以便查阅报错日志和后台输出
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# 建议使用目录模式 (COLL), 能够直接生成一个包含 exe 及其依赖 DLL 的文件夹
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ov-app',
)
