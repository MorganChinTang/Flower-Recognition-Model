# REAL TIME VIDEO PROCESSING - FLOWER RECOGNITION

# Overview
A **real-time flower classification system** using transfer learning with Vision Transformer (ViT-B/16). The project includes a Google Colab notebook for model training and development, and a desktop application for real-time video analysis with an interactive PyQt5 UI.

# Project Structure

## Desktop Application
- **Interactive UI**: PyQt5-based graphical interface with responsive design
- **Video Selection**: Dropdown menu to choose from available MP4 videos
- **Real-time Playback**: Display video frames with live flower predictions and confidence scores
- **Analysis Results**: 
  - Summary statistics (total frames, unique flowers, confidence metrics)
  - Results table with flower counts and average confidence per species
- **Multi-threaded Processing**: Responsive UI during video analysis
- **Auto-detection**: Automatically finds trained model and video files in project structure

## Installation

### Prerequisites
- Python 3.10.0 (I used 3.10.4)
- CUDA 11.8+ (optional, for GPU acceleration)
- 4GB+ RAM
- Visual Studio 2022

### Setup
**1. Make Sure the correct version of Python is installed**

**2. Clone the repository:**
```bash
git clone [https://github.com/MorganChinTang/VGP338_FlowerModelTraining.git](https://github.com/MorganChinTang/VGP338_FlowerModelTraining.git)
cd VGP338_FlowerModelTraining
```
**3. Install Dependencies:**

Copy and paste this into your Visual Studio 2022 terminal. 
```
cd c:\Users\user\Documents\GitHub\VGP338_FlowerModelTraining\FlowerModelTraining\FlowerModelTraining
pip install -r requirements.txt
python FlowerModelTraining.py
```
**4. Download Model:**
```
Go to "...\VGP338_FlowerModelTraining\FlowerModelTraining\TrainedModel"
Open 'Model_Download_Instruction.txt' And follow the instruction to obtain the trained model.
(The model was too big to be uploaded to git)
```
**5. All set! Now hit the start button on ur IDE**

## Google Colab Notebook
Open https://colab.research.google.com/drive/16WBT1zWwxYLf2Gj8zLBDPuuQDQHKU2-3?usp=drive_link and run all (Might take a while)
- **Dataset Management**: Download Oxford Flowers 102 dataset from Hugging Face
- **Progressive Fine-tuning**: 3-stage training approach
  - Stage 1: Train classification head only (5 epochs, lr=0.001)
  - Stage 2: Unfreeze last 3 transformer blocks (10 epochs, lr=0.0001)
  - Stage 3: Unfreeze all blocks (15 epochs, lr=0.00005)
- **Model Persistence**: Save/load models to Google Drive for cross-session training
- **Real-time Video Processing**: Process videos with frame-by-frame predictions
- **Pre-trained Model Integration**: Use `loretyan/vit-base-oxford-flowers-102` (99.51% accuracy)

## Model Information
- Architecture: Vision Transformer (ViT-B/16)
- Pre-training: ImageNet-21k
- Fine-tuning Dataset: Oxford Flowers 102 (8,189 training images, 1,020 test images)
- Classes: 102 flower species
- Model Size: ~344MB
- Training Strategy: Progressive unfreezing with decreasing learning rates

## Performance
- Final Model Accuracy: ~92.45% on Oxford Flowers 102
- Original Pre-trained Model Accuracy: 99.51% (loretyan/vit-base-oxford-flowers-102)
- Real-time Processing: ~1.1 FPS (every 5th frame analyzed)
- Device Support: NVIDIA CUDA (auto-detects, falls back to CPU)
- Average Confidence in flower type: 41.28% (Not enough flower types in dataset, would be a greater problem if the model is confident on a wrong answer)
- Max Confidence: 99.87%
- Min Confidence: 7.01%

## Demo Video
https://medal.tv/games/screen-capture/clips/lIVGAY676O1C1ZTtp?invite=cr-MSxnazMsMzMwNzEwMDY5 (Expires Dec 30th 2025)

## Google Slide (With Video and dodumentations)
https://docs.google.com/presentation/d/1yAY2KsX1n56hnZVH0lF-1JQhNrk65VgegzusUYsj4Jk/edit?usp=sharing
