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
