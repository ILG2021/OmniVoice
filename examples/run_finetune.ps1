# run_finetune.ps1
# This script demonstrates how to fine-tune OmniVoice from a JSONL manifest.
# PowerShell equivalent of run_finetune.sh for Windows.

#Requires -Version 5.1
$ErrorActionPreference = "Stop"

$stage      = 0
$stop_stage = 1

# ====== Modify as needed ======
# GPUs to use
$GPU_IDS  = "0"
$NUM_GPUS = 1

# Path to your input JSONL file
# (each line: {"id": ..., "audio_path": ..., "text": ..., "language_id": ...})
$TRAIN_JSONL = "data/train.jsonl"

# Path to your dev JSONL file. Set to empty string to skip dev set.
$DEV_JSONL = "data/dev.jsonl"

# Directory to write tokenized WebDataset shards
$TOKEN_DIR = "data/finetune/tokens"


# Training config file
# train_config_finetune_30h.json  – tuned for ~30h datasets (30k steps, recommended)
# train_config_finetune.json      – original 5k-step config (small datasets / quick tests)
# train_config_finetune_sdpa.json – use this if flex_attention fails on your GPU
$TRAIN_CONFIG = "config/train_config_finetune_30h.json"

# Data config file
$DATA_CONFIG = "config/data_config_finetune.json"

# Output directory for fine-tuned checkpoints
$OUTPUT_DIR = "exp/omnivoice_finetune"
# =================================

# Resolve the project root (parent of the examples/ folder) and add it to PYTHONPATH
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path

if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $ProjectRoot
}


# -----------------------------------------------------------------------
# Stage 0: Tokenize audio into WebDataset shards
# -----------------------------------------------------------------------
if ($stage -le 0 -and $stop_stage -ge 0) {
    Write-Host "Stage 0: Tokenizing audio"

    $splits = @(
        @{ jsonl = $TRAIN_JSONL; name = "train" }
        @{ jsonl = $DEV_JSONL;   name = "dev"   }
    )

    foreach ($s in $splits) {
        if ([string]::IsNullOrWhiteSpace($s.jsonl)) {
            continue
        }

        $split_name      = $s.name
        $split_jsonl_path = $s.jsonl
        $manifest        = "$TOKEN_DIR/$split_name/data.lst"

        if (Test-Path $manifest) {
            Write-Host "  Skipping $split_name – $manifest already exists."
            continue
        }

        Write-Host "  Tokenizing $split_name from $split_jsonl_path"

        $env:CUDA_VISIBLE_DEVICES = $GPU_IDS

        python -m omnivoice.scripts.extract_audio_tokens `
            --input_jsonl        $split_jsonl_path `
            --tar_output_pattern  "$TOKEN_DIR/$split_name/audios/shard-%06d.tar" `
            --jsonl_output_pattern "$TOKEN_DIR/$split_name/txts/shard-%06d.jsonl" `
            --nj_per_gpu         1 `
            --shuffle            True

        if ($LASTEXITCODE -ne 0) {
            Write-Error "Tokenization failed for split '$split_name'. Exit code: $LASTEXITCODE"
            exit $LASTEXITCODE
        }

        Write-Host "  Done. Manifest written to $manifest"
    }
}


# -----------------------------------------------------------------------
# Stage 1: Fine-tune
# -----------------------------------------------------------------------
if ($stage -le 1 -and $stop_stage -ge 1) {
    Write-Host "Stage 1: Fine-tuning"

    $env:CUDA_VISIBLE_DEVICES = $GPU_IDS

    accelerate launch `
        --gpu_ids      $GPU_IDS `
        --num_processes $NUM_GPUS `
        -m omnivoice.cli.train `
        --train_config $TRAIN_CONFIG `
        --data_config  $DATA_CONFIG `
        --output_dir   $OUTPUT_DIR

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Fine-tuning failed. Exit code: $LASTEXITCODE"
        exit $LASTEXITCODE
    }

    Write-Host "Fine-tuning complete. Checkpoints saved to: $OUTPUT_DIR"
}
