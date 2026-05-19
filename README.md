# DA6401 - Assignment 3: Transformer for Machine Translation

**GitHub Repo:** [Add link here]
**WandB Report:** [Add link here]

## Overview
This project implements the landmark "Attention Is All You Need" Transformer architecture from scratch in PyTorch. It trains a Neural Machine Translation (NMT) system to translate text from German to English using the Multi30k dataset.

## Key Features
- **Custom Architecture:** Complete implementation of Multi-Head Attention, Positional Encoding, and Encoder/Decoder stacks.
- **Optimization:** Utilizes the Noam Learning Rate Scheduler, Gradient Clipping, and Label Smoothing for stable and efficient training.
- **Decoding:** Implements greedy autoregressive decoding for inference and translation.
- **Evaluation:** Corpus-level BLEU score calculation for translation quality assessment.

## Core Files
- `model.py` - Core Transformer architecture (Encoders, Decoders, Attention)
- `dataset.py` - Multi30k dataset downloading, vocabulary building, and tokenization
- `train_modified.py` - Optimized training loop, checkpoint saving, and BLEU evaluation
- `lr_scheduler.py` - Noam learning rate scheduler implementation

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Build vocabulary and train the model: `python train_modified.py`
