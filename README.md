# Sign-Language-Recognition-with-CNN-Mamba
A deep learning pipeline for sign language recognition using CNN-based frame feature extraction and a Mamba sequence model for temporal understanding. Trained on the WLASL dataset.

This project implements a full end-to-end pipeline for American Sign Language (ASL) recognition using:

MediaPipe Pose + Hands keypoints extraction

Sequence preprocessing (pad/crop to a fixed length)

Temporal CNN layers

Mamba-style state-space layers for long-range temporal modeling

Attention pooling and a classifier head

Training on the WLASL dataset

The goal is to build a lightweight, efficient sign-language recognition system that works well in Colab or consumer hardware environments.

# Features
1. Automatic Keypoint Extraction

  Extracts for each frame:

  33 pose landmarks

  21 left-hand landmarks

  21 right-hand landmarks

  Each frame becomes a 150-dimensional vector. Missing hands are filled with zeros for consistency.

2. Preprocessing Pipeline

  For each video:

  Convert frames to keypoint vectors

  Save sequences as .npy files

  Pad or center-crop to a fixed sequence length (SEQ_LEN)

3. Clean Dataset Builder

  The dataset loader:

  Maps gloss names to labels

  Filters classes with only one sample

  Performs stratified train/val split

  Uses an efficient PyTorch Dataset and DataLoader

4. CNN + Mamba Architecture

  Model pipeline:

  Project 150 keypoints → D_MODEL

  Apply stacked temporal CNN layers

  Pass through Mamba-style temporal blocks

  Use attention pooling across the sequence

  Output class logits

5. Full Training Pipeline

  Includes:

  AdamW optimizer

  Cosine Annealing learning rate

  Label smoothing

  Gradient clipping

  Top-1 and Top-5 accuracy

  Automatic checkpoint saving

# Project Structure

Your dataset should look like:

AFML/
  Dataset/
    WLASL_Videos/
      about/
        1234.mp4
        1234.npy
      accept/
      accident/
      ...


Each folder is a gloss.
Each .npy file contains the extracted keypoint sequence for that video.

# Installation
pip install mediapipe opencv-python tqdm numpy
pip install torch torchvision torchaudio

# Keypoint Extraction

Set the index range depending on your Colab session:

START_IDX = 300
END_IDX   = 400


Then run:

python extract_keypoints.py

# Training

The training script includes:

--> Data loading

--> Sequence padding/cropping

--> CNN + Mamba model

--> Training loop

--> Validation loop

--> Checkpoint saving

Start training with:

python train.py


The best model is saved as:

cnn_mamba_wlasl_highacc.pt

# Model Architecture
Input: (B, T, 150) keypoint sequence
  ↓ Linear projection
  ↓ Temporal CNN layers (residual)
  ↓ Mamba-style temporal blocks
  ↓ Attention pooling
  ↓ Fully connected classifier
Output: (B, num_classes)

# Default Hyperparameters
Parameter	                   Value
SEQ_LEN	                      64
D_MODEL	                     192
CNN Layers	                  2
Mamba Layers	                2
Dropout	                     0.3
Learning Rate	              5e-4
Weight Decay	              5e-4
Label Smoothing	            0.1
Epochs	                     60
Batch Size	                 32
