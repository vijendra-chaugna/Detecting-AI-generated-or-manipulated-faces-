# Deepfake Detection with ResNet18 and MTCNN

A deep learning-based solution for detecting deepfake images and videos using face extraction with MTCNN and classification with ResNet18.

## 📋 Overview

This project implements a deepfake detection system that:
- Extracts faces from videos using MTCNN face detection
- Trains a ResNet18 classifier to distinguish between real and fake faces
- Provides both training and prediction capabilities
- Supports the FaceForensics++ dataset structure

## 🚀 Features

- **Face Extraction**: Automatic face detection and cropping from video frames
- **Deep Learning Model**: Fine-tuned ResNet18 for binary classification (Real vs Fake)
- **Flexible Input**: Support for images and videos
- **Multiple Compression Levels**: Compatible with FaceForensics++ compression types (raw, c23, c40)
- **GPU Support**: CUDA acceleration for faster training and inference

## 📁 Project Structure

```
├── deepfake_detector.py      # Main training and prediction script
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── data/                    # Processed dataset (created automatically)
    ├── real/                # Extracted real faces
    └── fake/                # Extracted fake faces
```

## 🛠 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd deepfake-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Dependencies
- torch
- torchvision
- facenet-pytorch
- opencv-python
- Pillow

## 📊 Dataset Preparation

This project is designed to work with the **FaceForensics++** dataset. Organize your dataset as follows:

```
FaceForensics/
├── original_sequences/
│   └── youtube/
│       └── [compression]/
│           └── videos/
└── manipulated_sequences/
    └── Deepfakes/
        └── [compression]/
            └── videos/
```

Supported compression levels: `raw`, `c23`, `c40`

## 🎯 Usage

### Training Mode

Train the deepfake detection model:

```bash
python deepfake_detector.py --mode train \
    --data ./FaceForensics \
    --compression raw \
    --epochs 10 \
    --batch-size 32 \
    --save-model best_model.pth
```

**Parameters:**
- `--data`: Path to FaceForensics++ dataset root
- `--compression`: Video compression level (raw/c23/c40)
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--save-model`: Output path for trained model

### Prediction Mode

Classify an image as real or fake:

```bash
python deepfake_detector.py --mode predict \
    --model best_model.pth \
    --image path/to/your/image.jpg
```

**Parameters:**
- `--model`: Path to trained model weights
- `--image`: Input image for classification

## 🔧 How It Works

### 1. Face Extraction
- Uses MTCNN for robust face detection
- Processes video frames with configurable frame skipping
- Saves cropped and resized face images (224×224)

### 2. Model Architecture
- **Backbone**: ResNet18 pre-trained on ImageNet
- **Classifier**: Custom binary classification head
- **Input**: 224×224 RGB face images
- **Output**: Real/Fake probability scores

### 3. Training Process
- Transfer learning from ImageNet weights
- Cross-entropy loss with Adam optimizer
- Automatic dataset preparation and face extraction
- Best model checkpointing

## 📈 Performance

The model achieves competitive performance on deepfake detection tasks:
- High accuracy on FaceForensics++ dataset
- Robust to various compression levels
- Fast inference with GPU acceleration

## ⚙️ Configuration

### Face Extraction Parameters
- **Frame Skip**: Process every 5th frame (configurable)
- **Confidence Threshold**: 0.70 for face detection
- **Face Size**: 224×224 pixels

### Training Parameters
- **Learning Rate**: 1e-4
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Validation**: Automatic train/validation split

## 🎮 Example Output

**Training:**
```
Extracting faces from videos...
Epoch 1/10, Loss: 0.4521
Epoch 2/10, Loss: 0.3215
...
Training complete. Best model saved to best_model.pth
```

**Prediction:**
```
Result: Fake (87.3%)
```

