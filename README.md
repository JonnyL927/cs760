# Recyclable Waste Recognition in Construction Sites Using Computer Vision Techniques

This project aims to recognize recyclable waste materials at construction sites using deep learning-based **semantic segmentation**. It addresses the lack of labeled data by adopting a **semi-supervised learning** approach and introduces several optimizations to improve performance in real-world scenarios.

## Overview

Construction and demolition waste (CDW) classification is crucial for promoting recycling and sustainable construction practices. Our framework segments images of construction sites to identify recyclable materials such as **timber, rubber, and brick**. Given the scarcity of annotated data, we adopt the **UniMatch semi-supervised segmentation** framework and propose several enhancements:

- Replacing encoder with more powerful **DINOv2** for richer visual representation.
- Apply a **unified data stream** to enhance the training efficiency.
- Introducing **Complementary Dropout** for better generalization.
- Using a **two-stage training strategy** to improve pseudo-label quality.

## Project Structure

```
UniMatch-V2/
├── configs/ # Configuration files (.yaml)
├── dataset/ # Data loading logic
├── docs/ # Project documentation
├── exp/ # Experiment logs and outputs
├── model/ # Network architecture definitions
├── preparing/ # Dataset pre-processing tools
├── pretrained/ # Pretrained weights (DINOv2 etc.)
├── scripts/ # Launchers for training/evaluation
├── splits/ # Dataset splits (train/val)
├── training-logs/ # Tensorboard logs and checkpoints
├── util/ # Utility scripts
├── distributed_train.py # Distributed training entry
├── fixmatch.py # FixMatch variant training
├── predict.py # Predict on single image
├── predict_video.py # Predict on video stream
├── supervised.py # Fully supervised training
├── test.py # Model evaluation
├── train.py # Main training pipeline
├── unimatch_v2.py # Enhanced UniMatch training logic
├── requirements.txt
└── README.md
```

## Getting Started

### Datasets

```
├── [Your CDW Path]
    ├── JPEGImages
    └── SegmentationClass
```

## Contribution

-   Gilda George led the data acquisition and preparation efforts: she annotated over 300 construction‐site images in Labelme, maintained detailed annotation guidelines (class labels, color maps, naming conventions), wrote Python scripts to convert JSON annotations into PNG masks, and organized the dataset into 60/20/20 train/validation/unlabeled splits. She also developed and documented example data‐loading pipelines in PyTorch, and co‐wrote the dataset description and preprocessing sections of the manuscript.

-   Qingyun Shi was responsible for pretraining model setup and baseline evaluation: he researched and selected candidate pretrained segmentation backbones (SegFormer, DeepLabV3+, Mask2Former), downloaded and integrated their weights into our inference pipeline, and ran initial supervised experiments on the labeled images to establish mIoU baselines. Qingyun prepared the baseline results figures and tables, performed comparative analysis across different encoders, and drafted the corresponding methodology and baseline evaluation portions of the report.

-   Junyi Liu implemented the semi‐supervised extension: he coded the UniMatch/FixMatch modules in PyTorch, including both weak‐to‐strong image augmentations and feature‐level perturbations, and generated pseudo‐labels with a 0.95 confidence threshold. Junyi logged per‐epoch pseudo‐label quality metrics in TensorBoard, performed ablation studies to measure the impact of complementary dropout, and wrote the sections describing semi‐supervised algorithms, loss functions, and performance comparisons.

-   Hejia Li took charge of fine‐tuning and evaluation: she defined the hyperparameter search space (learning rate, weight decay, dropout rate) and executed grid/manual tuning on the validation set with Optuna/scikit‐optimize. Hejia produced confusion matrices and class-IoU reports, measured inference FPS on GPU/CPU, visualized segmentati
