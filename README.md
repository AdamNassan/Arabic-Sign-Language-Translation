# Arabic Sign Language Translation 

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)](https://developer.nvidia.com/cuda-downloads)

A deep learning project for translating Arabic Sign Language to text using GRU-based encoder-decoder architecture with transfer learning from the Phoenix-2014T dataset.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Transfer Learning Process](#transfer-learning-process)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)

## 🔍 Overview

This project implements an Arabic Sign Language (ArSL) to text translation system using a GRU-based encoder-decoder architecture. The key innovation is the use of transfer learning from the Phoenix-2014T German sign language dataset to improve performance on the Arabic dataset.

### Key Achievements:
- **Transfer Learning**: Successfully implemented cross-dataset knowledge transfer
- **Architecture Optimization**: Fixed critical issues in bidirectional state handling
- **Performance Improvement**: Achieved faster convergence and better stability
- **Comprehensive Pipeline**: End-to-end solution from video processing to text output

## ✨ Features

- 🎯 **Transfer Learning**: Leverages Phoenix-2014T dataset for pretraining
- 🏗️ **Robust Architecture**: Bidirectional encoder with unidirectional decoder
- ⚡ **Efficient Training**: Differential learning rates and gradual unfreezing
- 📊 **Comprehensive Evaluation**: WER, BLEU scores, and training curves
- 🔧 **Flexible Pipeline**: Support for multiple feature extractors (Inception, VGG16, MobileNet)
- 📈 **Visualization**: Automatic plotting of training metrics

## 📁 Project Structure

```
EncoderDec/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.py                          # Device configuration
├── 
├── # Core Model Files
├── encoderDecoderModel.py             # Main model architectures
├── Encoder-Decoder.py                 # Training script for Arabic dataset
├── pretrain_on_phoenix.py             # Pretraining script for Phoenix dataset
├── attentionModel.py                  # Attention-based model (alternative)
├── 
├── # Data Processing
├── extract_frames.py                  # Video frame extraction
├── featureExtraction-main.py          # Feature extraction pipeline
├── featuresExtractor.py               # Feature extractor classes
├── models.py                          # Pretrained CNN models (Inception, VGG16, MobileNet)
├── reduce_features_dimension.py       # Dimension reduction (2048→1024)
├── generator.py                       # Dataset loaders
├── 
├── # Utilities
├── check_phoenix_format.py            # Phoenix dataset format checker
├── plot_metrics.py                    # Training metrics visualization
├── merge_csv_for_independent_new.py   # CSV file merging
├── utils.py                           # Utility functions
├── 
├── # Datasets
├── phoenix14t.pami0.train             # Phoenix training data
├── phoenix14t.pami0.dev               # Phoenix validation data
├── phoenix14t.pami0.test              # Phoenix test data
├── data/                              # Arabic dataset
│   ├── 01_train.csv, 01_test.csv     # Signer-specific data
│   ├── 02_train.csv, 02_test.csv
│   ├── ...
│   └── groundTruth.txt               # Vocabulary file
├── 
├── # Results and Checkpoints
├── checkpoints/
│   └── phoenix_best.pt               # Pretrained model weights
├── results/                          # Training results
├── Independent Results/              # Signer-independent results
├── Dependent Results/                # Signer-dependent results
└── PreTrained Independent Results/   # Transfer learning results
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory for training

### Setup Environment

1. **Clone the repository:**
```bash
git clone https://github.com/AdamNassan/Arabic-Sign-Language-Translation.git
cd Arabic-Sign-Language-Translation/ArabSign-EncoderDec/EncoderDec
```

2. **Create virtual environment:**
```bash
# Using conda (recommended)
conda create -n arabic-sign python=3.8
conda activate arabic-sign

# Or using venv
python -m venv arabic-sign
source arabic-sign/bin/activate  # Linux/Mac
# arabic-sign\Scripts\activate  # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify CUDA installation:**
```bash
python test_cuda.py
```

### Requirements.txt
```txt
torch>=1.9.0
torchvision>=0.10.0
torchaudio>=0.9.0
torchmetrics>=0.6.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
opencv-python>=4.5.0
pillow>=8.3.0
scikit-learn>=1.0.0
nltk>=3.6.0
```

## 📊 Dataset Preparation

### Phoenix-2014T Dataset (For Pretraining)

1. **Download Phoenix-2014T dataset:**
   - Visit: [Phoenix-2014T Dataset](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/)
   - Download the feature files (`.pkl.gz` format)
   - Place files in the root directory:
     - `phoenix14t.pami0.train`
     - `phoenix14t.pami0.dev`
     - `phoenix14t.pami0.test`

2. **Verify dataset format:**
```bash
python check_phoenix_format.py phoenix14t.pami0.train
```

### Arabic Dataset

1. **Prepare video data:**
   - Organize videos in the structure: `signer/split/class/sample.mp4`
   - Example: `01/train/hello/sample_001.mp4`

2. **Extract frames:**
```bash
python extract_frames.py
```

3. **Extract features:**
```bash
python featureExtraction-main.py
```

4. **Reduce feature dimensions (if using Inception):**
```bash
python reduce_features_dimension.py
```

5. **Generate CSV files:**
   - Create CSV files mapping features to captions
   - Format: `index,sentId,sPath,framesNo,signerID,caption,procCaption`

## 🎯 Usage

### Quick Start

1. **Pretrain on Phoenix dataset:**
```bash
python pretrain_on_phoenix.py
```

2. **Train on Arabic dataset with transfer learning:**
```bash
python Encoder-Decoder.py
```

3. **Visualize training metrics:**
```bash
python plot_metrics.py
```

### Detailed Training Process

#### Phase 1: Pretraining on Phoenix Dataset

```bash
# Configure device and hyperparameters in config.py
python config.py

# Start pretraining
python pretrain_on_phoenix.py
```

**Pretraining Configuration:**
- Batch size: 32
- Learning rate: 0.0003
- Epochs: 50
- Hidden size: 256
- Dropout: 0.7
- Optimizer: AdamW with weight decay

#### Phase 2: Transfer Learning on Arabic Dataset

```bash
# Modify paths in Encoder-Decoder.py if needed
python Encoder-Decoder.py
```

**Transfer Learning Strategy:**
- Load pretrained encoder weights from `checkpoints/phoenix_best.pt`
- Freeze encoder layers initially
- Use differential learning rates (encoder: 1e-4, decoder: 1e-3)
- Gradually unfreeze encoder after N epochs

## 🔄 Transfer Learning Process

### Step-by-Step Process

1. **Model Architecture Preparation:**
   ```python
   # Create encoder with pretrained weights capability
   encoder = EncDecModel.EncoderRNN(
       input_size=1024,
       hidden_size=256,
       num_layers=2,
       dropout=0.7
   )
   
   # Create decoder for Arabic vocabulary
   decoder = EncDecModel.DecoderRNN(
       num_embeddings=arabic_vocab_size,
       embedding_dim=300,
       input_size=300,
       hidden_size=256,
       num_layers=2,
       dropout=0.7,
       vocab_size=arabic_vocab_size
   )
   ```

2. **Load Pretrained Weights:**
   ```python
   def load_pretrained_weights(model, checkpoint_path):
       checkpoint = torch.load(checkpoint_path)
       model.load_state_dict(checkpoint['model_state_dict'], strict=False)
       return model
   ```

3. **Apply Transfer Learning Strategy:**
   ```python
   # Freeze encoder layers
   freeze_encoder_layers(model.encoder)
   
   # Use differential learning rates
   optimizer = get_optimizer_with_different_lrs(
       model, encoder_lr=1e-4, decoder_lr=1e-3
   )
   
   # Gradual unfreezing
   if epoch >= unfreeze_epoch:
       unfreeze_encoder_layers(model.encoder)
   ```

## 🏗️ Model Architecture

### Architecture Overview

```
Input Video Frames (80 frames × 1024 features)
                    ↓
            Bidirectional GRU Encoder
         (2 layers, hidden_size=256)
                    ↓
           Feature Representation
                    ↓
            Unidirectional GRU Decoder
         (2 layers, hidden_size=512)
                    ↓
            Output Arabic Text
```

### Key Components

1. **EncoderRNN (Bidirectional):**
   - Processes temporal features from video frames
   - Captures forward and backward dependencies
   - Outputs concatenated bidirectional states

2. **DecoderRNN (Unidirectional):**
   - Generates Arabic text sequences
   - Uses teacher forcing during training
   - Handles variable-length outputs

3. **Enhanced Architecture (EncoderRNNPre/DecoderRNNPre):**
   - Improved bidirectional state handling
   - Proper dimension management
   - Transfer learning compatibility

### Architecture Differences

| Component | Original | Enhanced (Pre-training) |
|-----------|----------|------------------------|
| Encoder States | Summed bidirectional | Concatenated bidirectional |
| Decoder Direction | Bidirectional (incorrect) | Unidirectional (correct) |
| Hidden Size | 256 | 512 (to match encoder) |
| State Handling | Basic | Advanced dimension management |

## 🎓 Training

### Training Configuration

```python
# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
NUM_EPOCHS = 100
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.7
TEACHER_FORCING_RATIO = 0.5

# Transfer Learning
ENCODER_LR = 1e-4
DECODER_LR = 1e-3
UNFREEZE_EPOCH = 10
```

### Training Process

1. **Phase 1 - Pretraining:**
   - Dataset: Phoenix-2014T (7,096 sequences)
   - Objective: Learn general sign language features
   - Duration: ~50 epochs until convergence

2. **Phase 2 - Fine-tuning:**
   - Dataset: Arabic Sign Language
   - Strategy: Transfer learning with frozen encoder
   - Duration: ~30-40 epochs

### Training Script Usage

```bash
# Basic training
python Encoder-Decoder.py

# With custom parameters
python Encoder-Decoder.py --batch_size 64 --learning_rate 0.001

# Resume from checkpoint
python Encoder-Decoder.py --resume checkpoints/model_epoch_20.pt
```

## 📈 Evaluation

### Metrics

1. **Word Error Rate (WER):**
   - Primary evaluation metric
   - Measures sequence-level accuracy
   - Lower is better

2. **BLEU Scores:**
   - BLEU-1, BLEU-2, BLEU-3, BLEU-4
   - Measures n-gram overlap
   - Higher is better

3. **Training Loss:**
   - CrossEntropyLoss with padding mask
   - Tracked during training

### Evaluation Script

```python
# Evaluate model
def eval_model(test_loader, model, captionToIndex, indexToCaption, 
               modelName, vocab_size, printPredictions=False):
    model.eval()
    total_wer = 0
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Generate predictions
            outputs = model(batch['features'], batch['targets'])
            # Calculate WER and BLEU scores
            # ...
    
    return wer, bleu_scores, predictions
```

### Results Visualization

```bash
# Plot training curves
python plot_metrics.py

# Generate result tables
python generate_results_table.py
```

## 📊 Results

### Performance Comparison
#### 1- Signer Dependent
| Model | Training Time | Convergence | Final WER | Improvement |
|-------|---------------|-------------|-----------|-------------|
| Baseline GRU | ~1 hour | 50 epochs | 0.07% | - |
#### 2- Signer Independent

| Model | Training Time | Convergence | Final WER | Improvement |
|-------|---------------|-------------|-----------|-------------|
| Baseline GRU | ~6 hours | 50 epochs | 0.71% | - |
| With Transfer Learning | ~4.5 hours | 30 epochs | 0.65% | 0.06% better |

### Training Curves

The `plot_metrics.py` script generates training curves showing:
- Training vs Validation WER over epochs
- Loss convergence patterns
- Transfer learning impact visualization

### Key Findings

1. **Transfer Learning Benefits:**
   - 40-50% faster convergence
   - Improved stability during training
   - Better final performance

2. **Architecture Improvements:**
   - Fixed bidirectional state handling
   - Proper tensor dimension management
   - Enhanced gradient flow

3. **Training Strategy:**
   - Differential learning rates proved effective
   - Gradual unfreezing prevented catastrophic forgetting
   - Teacher forcing ratio scheduling improved results

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory:**
   ```python
   # Reduce batch size
   BATCH_SIZE = 16  # Instead of 32
   
   # Clear cache
   torch.cuda.empty_cache()
   ```

2. **Dimension Mismatch Errors:**
   ```python
   # Check tensor shapes
   print(f"Features shape: {features.shape}")
   print(f"Hidden shape: {hidden.shape}")
   
   # Ensure proper data loading
   features = features.permute(1, 0, 2)  # (seq_len, batch, features)
   ```

3. **Phoenix Dataset Loading Issues:**
   ```bash
   # Verify file format
   python check_phoenix_format.py phoenix14t.pami0.train
   
   # Check gzip compression
   file phoenix14t.pami0.train
   ```

4. **Training Instability:**
   ```python
   # Add gradient clipping
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   
   # Increase dropout
   dropout = 0.8
   
   # Use learning rate scheduling
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
   ```


### 📈 Project Status

- ✅ Core implementation complete
- ✅ Transfer learning implemented
- ✅ Evaluation metrics added
- ✅ Documentation complete
- 🔄 Performance optimization ongoing
- 📋 Future: Real-time inference pipeline

### 🔮 Future Work

- [ ] Real-time video processing
- [ ] Mobile app deployment
- [ ] Attention mechanism enhancement
- [ ] Multi-modal fusion (RGB + depth)
- [ ] Transformer architecture adaptation
- [ ] Cross-lingual transfer learning

---

