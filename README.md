# MNIST Classifier Project

This project implements three different classification models for the MNIST dataset using Object-Oriented Programming principles in Python. The implementation includes Random Forest, Feed-Forward Neural Network, and Convolutional Neural Network classifiers under a unified interface.

## Project Structure

```
/test_task_1
│── /models
│   │── __init__.py
│   │── base.py
│   │── cnn.py
│   │── feed_forward.py
│   │── random_forest.py
│── /interfaces
│   │── mnist_classifier_interface.py
│── /utils
│   │── __init__.py
│   │── dataset.py
│   │── metrics.py
│   │── visualization.py
│── classifier.py
│── main.py
│── README.md
```

## Features

- Three different classification models:
  - Random Forest Classifier
  - Feed-Forward Neural Network
  - Convolutional Neural Network
- Unified interface for all models
- Automated MNIST dataset downloading and preprocessing
- Model training and evaluation capabilities
- Easy model switching through a single parameter

## Requirements

```bash
Python==3.10.8
matplotlib==3.9.1
numpy==1.26.4
scikit_learn==1.6.1
seaborn==0.13.2
torch==2.3.1
torchvision==0.18.1
```

## Installation

1. Create a virtual environment and activate it:
```bash
python -m venv ml_env
source ml_env/bin/activate  # For Linux/Mac
ml_env\Scripts\activate  # For Windows
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage example:

```python
from classifier import MnistClassifier
from utils.dataset import MnistDataset

# Get MNIST dataset
mnist_dataset = MnistDataset(batch_size=32)
train_loader, test_loader = mnist_dataset.get_data_loaders()

# Create and train model
# algorithm can be 'cnn', 'rf', or 'nn'
classifier = MnistClassifier(algorithm='nn')
classifier.train(train_loader, epochs=5)

# Make predictions
predictions = classifier.predict(test_loader)

# Evaluate model
accuracy = classifier.evaluate(test_loader)
print(f"Test Accuracy: {accuracy:.2f}%")
```

## Model Details

### Random Forest
- Uses scikit-learn's RandomForestClassifier
- Default 100 estimators
- Suitable for smaller datasets

### Feed-Forward Neural Network
- Two-layer neural network
- ReLU activation
- Dropout for regularization
- Adam optimizer

### Convolutional Neural Network
- Two convolutional layers
- Max pooling
- ReLU activation
- Fully connected layers
- Adam optimizer

## Performance

Typical performance metrics on MNIST test set:
- Feed-Forward NN: ~97-98% accuracy
- CNN: ~98-99% accuracy
- Random Forest: ~96-97% accuracy

## Project Structure Details

- `interfaces/mnist_classifier_interface.py`: Defines the abstract interface for all classifiers
- `models/`: Contains implementation of all three classifiers
- `utils/`: Contains helper functions for data loading and processing
- `classifier.py`: Main classifier class that provides a unified interface
- `main.py`: Example usage script

## Contributing

Feel free to submit issues and enhancement requests!

## Note

This is a test assignment for further learning purposes.

