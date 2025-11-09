# Food Image Classification with Custom CNN

This project builds, trains, and evaluates a Convolutional Neural Network (CNN) for food image classification using a subset of the Food-101 dataset. The goal is to achieve robust accuracy on a small, scalable subset for eventual extension to the full dataset.

## Project Overview

**Objective:**  
Classify images of food (pizza, steak, sushi) using a CNN and achieve at least 80% classification accuracy on the test set.

**Dataset:**  
Food-101, using a subset of 2,000 training and 700 test images split across 3 classes for efficiency and speed.

**Performance Goal:**  
≥80% accuracy on test data.

**Constraints:**  
Designed to fit within 15 GB storage and run in under 2 hours on CPU (faster on GPU).

## Data Pipeline

- Dataset: Subset of Food-101 (pizza, steak, sushi).  
- Size: 2,000 training, 700 test samples, resized to 224x224 pixels.  
- Augmentation: Random horizontal/vertical flip, rotation (±15°), color jitter, affine transforms.  
- Normalization: ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.  
- Split: Official splits from the dataset; batch size of 32.

## Model Architecture

- Type: Custom CNN  
- Blocks: 4 convolutional blocks (filters: 32, 64, 128, 256), batch normalization, max pooling, dropout (0.15).  
- Heads: 2 fully connected layers (sizes: 512, 256) with dropout (0.5).  
- Activation: ReLU throughout, global average pooling before FC layers.  
- Parameters: ~1.46 million.

## Training & Optimization

- Optimizer: Adam with weight decay (1e-4).  
- Learning Rate: 0.001 (ReduceLROnPlateau scheduler, factor 0.5 on plateau, patience 3, min LR 1e-6).  
- Loss: CrossEntropyLoss.  
- Epochs: Up to 15, early stopping with patience 5.  
- Regularization: Weight decay, dropout, gradient clipping (max norm 1.0).

## Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 80.3%  |
| Precision | 80.5%  |
| Recall    | 80.3%  |
| F1-Score  | 80.3%  |

See `training_history.jpg` for plots of loss, accuracy, learning rate, and final scores.

Training and validation loss both decrease and stabilize across epochs, with minimal overfitting.  
Training and validation accuracy closely track each other, converging above 80%.

## How to Use

1. **Requirements:** Python, PyTorch, torchvision, scikit-learn, matplotlib, ultralytics.  
2. **Dataset:** Download Food-101 via torchvision or from Kaggle, or use provided subset scripts.  
3. **Run Training:** Execute the notebook `Lab08.ipynb` step by step.  
4. **Visualizations:** Training metrics and model performance plots are saved as `training_history.jpg` for review.  
5. **Inference:** Use the last notebook cells to test custom images.

## Files

- `Lab08.ipynb`: Jupyter notebook containing code, experiments, and results.  
- `training_history.jpg`: Visualization of model metrics and performance.  
- `FINAL_REPORT_1.txt`: Detailed markdown report, including architecture, configuration, and insights.

## Insights

- Batch normalization and data augmentation were crucial for achieving stable and high accuracy.  
- Learning rate scheduling dramatically improved convergence.  
- Standard data pipeline and CNN principles remain effective on small and balanced class subsets.  
- Early stopping and dropout prevented overfitting even on a small data subset, enabling later scaling to more classes and data.

---
