# COMP0087 Stage 1

This repository is for Stage 1 of the COMP0087 group project.

## Goal
Build a runnable Stage 1 SFT pipeline that can:

- read unified data
- preprocess data
- initialize training
- save checkpoint
- run minimal inference check

## Structure

- `train_sft.py`: main training entry
- `configs/debug.yaml`: debug config
- `src/data/`: data loading
- `src/preprocessing/`: preprocessing
- `src/training/`: trainer building
- `src/utils/`: config and utilities
- `data/debug/`: debug samples
- `outputs/`: checkpoints and outputs

## Current status

Initial repository skeleton for Stage 1 integration.





## Member 6: Configs, Run Scripts, and Minimal Verification

This branch adds the Stage 1 helper pipeline for configuration management, smoke testing, and minimal post-training verification.

### 1. Run the debug smoke pipeline

Use the debug script to run the smallest end-to-end pipeline once:

```bash
bash scripts/run_debug.sh