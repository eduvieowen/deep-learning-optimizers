# Deep Learning Optimizers

This project implements a feedforward neural network using PyTorch to classify images from the KMNIST dataset. The training process utilizes 5-fold cross-validation to ensure robust evaluation and compares the performance of three optimizers: Adam, RMSprop, and AdamW.

## Features

* Data Loading: Automatically downloads and loads the KMNIST dataset.
* Model Architecture: A simple feedforward network with two hidden layers.
* Cross-Validation: Uses 5-fold cross-validation to assess model generalization.
* Optimizer Comparison: Evaluates different optimizers by tracking training loss, validation loss, training accuracy, and validation accuracy.
* Visualization: Generates plots for loss and accuracy trends across training epochs.
* Results Summary: Creates a summary table including average metrics and training time.

## Requirements

* Python 3.x
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Pandas
* Scikit-learn

## Model Architecture

* Input Layer: Accepts a flattened image of size 28*28 (784 features).
* First Hidden Layer: 128 neurons.
* Second Hidden Layer: 64 neurons.
* Output Layer: 10 neurons corresponding to the 10 classes.

## Hyperparameters

The following hyperparameters can be adjusted to fine-tune the training process:

* `batch_size`: Number of samples processed in each training batch. By default it is set to 64.
* `epochs`: Number of training epochs per cross-validation fold. By default it is set to 10
* `learning_rate`: The learning rate used by the optimizer. By default it is set to 0.001
* Optimizer Choices: The script supports three optimizers: Adam, RMSprop, AdamW. You can modify the "optimizer_name", "optimizer" and "optimizers" variables in the code to experiment with these or add new ones.
* K-Fold Cross-Validation: `k_folds`: Number of folds for cross-validation. By default it is set to 5