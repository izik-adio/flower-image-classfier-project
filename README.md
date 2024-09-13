# Image Classifier Project

This project is the final submission for the AI Programming with Python Nanodegree. It demonstrates how to build and train an image classifier using deep learning with PyTorch, as well as a command-line application to predict the class of an image.

## Project Overview

The project is divided into two parts:

1. **Part 1: Development Notebook**
   - Implements data preprocessing, model creation, training, validation, and saving.
2. **Part 2: Command-Line Application**
   - Implements a script to train and save models, as well as a prediction script that identifies image classes using the trained model.

## Installation

### Dependencies

Ensure that the following libraries are installed:

- `torch`
- `torchvision`
- `PIL`
- `matplotlib`
- `json`
- `argparse`

These can be installed with:

```bash
pip install torch torchvision matplotlib pillow
```

## Part 1: Development Notebook

The development notebook (`final_01/final-01/Image Classifier Project.ipynb`) covers the following steps:

### 1. Data Preprocessing

- **Package Imports**: Imports necessary libraries, including PyTorch and torchvision.
- **Data Augmentation**: Utilizes `torchvision.transforms` to augment training data with random scaling, rotations, mirroring, and cropping.
- **Normalization**: Applies data normalization for all datasets (training, validation, and testing).
- **Data Batching**: Uses `torchvision.datasets.ImageFolder` and `DataLoader` to load datasets in batches.

### 2. Model Architecture

- **Pretrained Network**: Loads a pretrained EffieicentNet_B0 model from `torchvision.models` and freezes its parameters.
- **Feedforward Classifier**: Defines a new classifier for transfer learning on top of the EffieicentNet_B0 feature extractor.

### 3. Model Training and Evaluation

- **Training**: Trains the new classifier while keeping the feature network parameters static.
- **Validation**: Displays validation loss and accuracy during training.
- **Testing**: Measures the model’s accuracy on test data.

### 4. Checkpoints

- **Saving the Model**: Saves the trained model’s checkpoint, including hyperparameters and class mapping.
- **Loading the Model**: Implements a function to reload the saved model from a checkpoint.

### 5. Image Processing and Class Prediction

- **Image Processing**: A function (`process_image`) converts a PIL image into a format usable by the trained model.
- **Prediction**: A function (`predict`) returns the top K most probable classes for a given image.
- **Sanity Check**: Uses `matplotlib` to display the image and its top 5 predicted classes, including actual class names.

## Part 2: Command Line Application

Two Main scripts are provided in this section to allow users to train a model and make predictions using the command line.

### 1. `train.py`

- **Training a Model**: Allows training a new network on a dataset and saving it as a checkpoint.
- **Model Architecture**: Users can choose between three different models `Resnet, VGG, EffieicentNet`.
- **Hyperparameter Tuning**: Users can set the learning rate, number of hidden units, and the number of epochs.
- **Training on GPU**: Users can specify whether to train on a GPU.

Usage:

```bash
python final-02\train.py --data_dir <path_to_data> --save_dir <path_to_checkpoint> --arch <model_name> --learning_rate <lr> --hidden_units <units> --epochs <epochs> --gpu
```

### 2. `predict.py`

- **Predicting Classes**: Loads an image and a saved checkpoint to predict the class of the image.
- **Top K Classes**: Allows users to print the top K most likely classes and their probabilities.
- **Class Names**: Users can load a JSON file that maps class indices to real-world labels.
- **Prediction on GPU**: Users can choose to use a GPU for predictions.

Usage:

```bash
python final-02\predict.py --image_path <image_path> --checkpoint <checkpoint_path> --top_k <K> --category_names <json_file> --gpu
```

## Results

The model was trained on a flower classification dataset, achieving a high accuracy on the test set. The command-line application allows users to easily train new models and make predictions with pre-trained models.

## License

This project is licensed under the MIT License.
