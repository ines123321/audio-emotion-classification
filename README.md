#  Audio Emotion Classification with ResNet50 and Diffusion Models

**Research Context**

This project is inspired by and aims to reproduce key aspects of:

**"A Generation of Enhanced Data by Variational Autoencoders and Diffusion Modeling"**
- *Authors*: Young-Jun Kim and Seok-Pil Lee
- *Journal*: Electronics 2024, 13(7), 1314
- *DOI*: [https://doi.org/10.3390/electronics13071314](https://doi.org/10.3390/electronics13071314)

**Key concepts reproduced from the paper:**
-  Use of diffusion models for emotional speech data augmentation
-  Mel-spectrogram representation of audio
-  ResNet-based emotion recognition
-  EmoDB and RAVDESS datasets

**Differences from the original paper:**
-  Simplified diffusion architecture (Compact U-Net vs full architecture)
-  Fewer training epochs (50 vs 800)
-  Focus on practical implementation for limited resources


---

##  Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
- [Methodology](#methodology)
---

##  Overview

Speech emotion recognition (SER) is a crucial task in affective computing with applications in human-computer interaction, mental health monitoring, and customer service. This project explores:

1. **Transfer Learning**: Using pre-trained ResNet50 on mel-spectrograms
2. **Data Augmentation**: Traditional time/frequency masking techniques
3. **Generative Augmentation**: Diffusion models conditioned on emotion embeddings
4. **Performance Analysis**: Comparing baseline vs augmented approaches

### Problem Statement
Audio emotion classification suffers from:
- **Limited labeled data** (1,510 samples in our case)
- **Class imbalance** across emotion categories
- **High overfitting** due to small dataset size

### Our Solution
Generate synthetic mel-spectrograms using diffusion models conditioned on emotion-specific embeddings from trained ResNet50, increasing dataset size by **159%** (1,510 → 3,910 samples).

---

##  Results

| Approach | Dataset Size | Val Acc | Improvement |
|----------|--------------|---------|-------------|
| **Baseline** (ResNet50) | 1,510 | 74% | - |
| **Traditional Augmentation** | 1,510 | 79% | +3% |
| **Diffusion Augmentation** | 3,910 | 84% | **+5%** |

### Key Findings
 **Synthetic data improves generalization** by 5%  
 **Reduced overfitting**gap  
 **Limitation**: Generated samples remain similar to originals, limiting diversity gains

---

##  Key Features

-  **Multi-dataset support**: EmoDB (German) + RAVDESS (English)
-  **Transfer learning**: Pre-trained ResNet50 on ImageNet
-  **Mel-spectrogram preprocessing**: 80 mel bands, 224×224 resolution
-  **Advanced augmentation**: Time/frequency masking, noise injection
-  **Diffusion-based generation**: Emotion-conditioned synthetic data
-  **Comprehensive evaluation**: Confusion matrices, per-class metrics
-  **Efficient caching**: Pre-computed mel-spectrograms for fast training

---

##  Project Architecture

```
audio-emotion-classification/
│
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── notebooks/
│   └── audio_emotion_classification.ipynb  # Main notebook with full pipeline
│
├── data/                        # Raw datasets (NOT in Git)
│   ├── .gitkeep
│   ├── emodb/                   # Berlin Emotional Speech DB
│   │   ├── .gitkeep
│   │   └── *.wav (535 files)
│   └── ravdess/                 # RAVDESS Dataset
│       ├── .gitkeep
│       ├── Actor_01/
│       └── ... (24 actors, 1440 files)
│
├── models/                      # Trained models 
│   ├── .gitkeep
│   ├── best_resnet_model.pth
│   ├── emotion_recognition_complete.pth
│   ├── diffusion_model.pth
│   ├── emotion_embeddings.pth
│   └── emotion_centroids.pth
│
├── cache/                       # Cached data 
│   ├── .gitkeep
│   ├── melspectrogram_cache/
│   │   ├── .gitkeep
│   │   └── *.npy (cached spectrograms)
│   └── synthetic_data/
│       ├── .gitkeep
│       └── synthetic_emotion_data_diffusion_COMBINED.pth
│
├── sample_results/              # Visualizations 
│   ├── .gitkeep
│   ├── confusion_matrix_traditional_Data_Augmentation.png
│   └── confusion_matrix_augmented_data_with_diffusion.png
│
└── configs/                     # Configuration files (NOT in Git)
    ├── .gitkeep
    ├── dataset_metadata.pkl
    └── global_config.pkl
```

---

##  Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU works)
- 8GB RAM minimum (16GB recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/audio-emotion-classification.git
cd audio-emotion-classification
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download datasets** (see [Datasets](#datasets) section)

---

##  Datasets

### EmoDB (Berlin Database of Emotional Speech)

- **Kaggle Dataset:** [https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb](https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb)
- **Official Source:** [http://emodb.bilderbar.info/](http://emodb.bilderbar.info/)



**Description**: German emotional speech database with 535 utterances from 10 actors (5 male, 5 female).


**Emotions**: 7 classes
- Anger (Wut) - W
- Boredom (Langeweile) - L
- Disgust (Ekel) - E
- Fear (Angst) - A
- Happiness (Freude) - F
- Sadness (Trauer) - T
- Neutral - N

**File naming**: `[Speaker][Text][Emotion][Version].wav`
- Example: `03a01Fa.wav` = Speaker 03, Text a01, Emotion F (happiness)

---

### RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

- **Kaggle Dataset:** [https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- **Official Source (Zenodo):** [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)

**Description**: English emotional speech and song database with 1,440 audio files from 24 actors (12 male, 12 female).


**Emotions**: 8 classes
- Neutral (01)
- Calm (02)
- Happiness (03)
- Sadness (04)
- Anger (05)
- Fear (06)
- Disgust (07)
- Surprise (08)

**File naming**: `Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav`
- Example: `03-01-06-01-02-01-12.wav`

---

### Dataset Preparation

After downloading both datasets, your structure should look like:

```
data/
├── emodb/
│   ├── 03a01Fa.wav
│   ├── 03a01Wa.wav
│   └── ... (535 files)
└── ravdess/
    ├── Actor_01/
    │   ├── 03-01-01-01-01-01-01.wav
    │   └── ...
    └── ... (24 actor folders)
```

**Common emotions used**: For this project, we use 6 emotions present in both datasets:
- Anger
- Sadness
- Fear
- Happiness
- Neutral
- Disgust

---

##  Usage

### Quick Start

1. **Open Jupyter Notebook**
```bash
jupyter notebook notebooks/audio_emotion_classification.ipynb
```

2. **Run all cells sequentially** - The notebook is organized as follows:

#### Section 1-3: Data Loading and Preprocessing
- Load EmoDB and RAVDESS datasets
- Extract mel-spectrograms
- Create PyTorch datasets

#### Section 4-6: Baseline Training
- Train ResNet50 on original data
- Evaluate baseline performance 

#### Section 7-9: Traditional Augmentation
- Apply time/frequency masking
- Retrain with augmented data 

#### Section 10-14: Diffusion Model Training
- Extract emotion embeddings from ResNet
- Train diffusion model for synthetic generation
- Generate 1,200+ synthetic samples

#### Section 15-17: Final Training with Synthetic Data
- Combine original + synthetic data (3,910 total)
- Retrain ResNet50
- Achieve **84% accuracy**

#### Section 18: Evaluation
- Confusion matrices
- Per-class accuracy
- Overfitting analysis

---

### Google Colab Usage

For Colab users:

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Upload datasets to Drive
# /content/drive/MyDrive/emodb/
# /content/drive/MyDrive/ravdess/

# 3. Copy to Colab runtime
!cp -r /content/drive/MyDrive/emodb /content
!cp -r /content/drive/MyDrive/ravdess /content

# 4. Run notebook cells
```

---

##  Methodology

### 1. Audio Preprocessing

**Input**: Raw audio waveforms (.wav files)

**Process**:
1. Load audio at 22,050 Hz sample rate
2. Pad/truncate to 10 seconds
3. Extract mel-spectrogram (80 mel bands)
4. Convert to dB scale
5. Normalize (z-score)
6. Resize to 224×224 for ResNet

**Output**: 3-channel mel-spectrogram (224×224×3)

---

### 2. Baseline Model: ResNet50

**Architecture**:
- Pre-trained ResNet50 (ImageNet weights)
- Fine-tune last layers
- Replace FC layer for 6 emotion classes
- Add dropout (0.5) for regularization

**Training**:
- Optimizer: Adam (lr=1e-4)
- Loss: CrossEntropyLoss
- Scheduler: CosineAnnealingLR
- Epochs: 50 with early stopping

---

### 3. Traditional Data Augmentation

**Techniques**:
- **Time Masking**: Mask random time segments (max 10%)
- **Frequency Masking**: Mask random frequency bands (max 15%)
- **Time Shifting**: Shift spectrogram in time (±10%)
- **Gaussian Noise**: Add random noise (std=0.01)
- **Random Scaling**: Scale amplitude (0.9-1.1x)

**Application**: On-the-fly during training with 70% probability

---

### 4. Diffusion-Based Synthetic Generation

**Step 1: Extract Emotion Embeddings**
- Use trained ResNet50 (before FC layer)
- Extract 2048-dim embeddings for all samples
- Compute emotion centroids (average per class)

**Step 2: Train Diffusion Model**
- Architecture: Compact U-Net with attention
- Conditioning: Emotion embeddings from ResNet
- Process: Forward diffusion (add noise) + Reverse (denoise)
- Timesteps: 50 (optimized for speed)

**Step 3: Generate Synthetic Data**
- Input: Emotion centroid + random variation
- Process: Start from noise, denoise 50 steps
- Output: Synthetic mel-spectrogram
- Generate: 200 samples per emotion (1,200 total)

**Step 4: Combine and Retrain**
- Merge original (1,510) + synthetic (1,200) = 3,910 samples
- Stratified train/val split (80/20)
- Retrain ResNet50 from scratch

---



## Experimental Insights

### What Worked 
1. **Transfer learning** from ImageNet significantly boosted performance
2. **Mel-spectrograms** as 2D images work well with CNNs
3. **Diffusion models** generate realistic emotional speech patterns
4. **Emotion conditioning** via ResNet embeddings enables targeted generation

### What Didn't Work 
1. **Massive synthetic generation** (3,000+ samples) didn't further improve accuracy
2. **Similar synthetic samples** limit diversity gains
3. **Overfitting persists** even with 3× dataset size

### Limitations 
1. **Dataset bias**: EmoDB (German) + RAVDESS (English) - language mixing
2. **Limited actors**: Only 34 speakers total (10 EmoDB + 24 RAVDESS)
3. **Synthetic similarity**: Generated data too close to training distribution
4. **Computational cost**: Diffusion generation takes ~2 hours for 1,200 samples


