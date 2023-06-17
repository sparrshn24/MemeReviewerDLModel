# MemeReviewerDLModel

This Python script is designed to perform data preprocessing and exploration for a deep learning model in the context of image classification. It includes various imports, installations, and setup steps specific to the Google Colab environment.

## Prerequisites
- Python 3.x
- TensorFlow library
- Keras library
- Scipy library
- NumPy library
- Seaborn library
- Keras Tuner library
- Scikit-learn library
- Matplotlib library
- TensorFlow Addons library
- TensorFlow Hub library
- TensorFlow Text library
- Official TensorFlow Models library

## Getting Started
1. Install the required dependencies by running the necessary `pip install` commands in your command line.
2. Make sure you have access to the necessary data files: `training_set_task3.zip` and `dev_set_task3.zip`.
3. Update the file paths in the script to point to the correct locations of the data files.
4. Run the script using Python.

## Script Overview
1. The script installs and imports the necessary libraries and dependencies for the deep learning model.
2. It sets up the necessary environment configurations, such as TensorBoard logging.
3. The script defines utility functions for plotting learning curves and creating callbacks for model training.
4. It performs data exploration by loading and examining the training and validation data.
5. The script preprocesses the data, including checking for special characters, handling image files, and performing sanity checks on the data.
6. It encodes the labels using a dictionary mapping technique names to numerical values.
7. The script cleans the text data by removing blank spaces and newline characters.
8. Finally, the script prepares the data for further processing or model training.
   
# Data Generator

This script provides a `DataGenerator` class that generates data for Keras models. It is designed to handle image and text data, along with corresponding labels. The `DataGenerator` class is a subclass of `keras.utils.Sequence` and provides methods for data loading, preprocessing, and batching.

## Usage

1. Import the required modules:

```python
from itertools import filterfalse
from warnings import filterwarnings
```

2. Instantiate a `DataGenerator` object:

```python
training_gen = DataGenerator(df_json, n_classes=22, data_prefix=training_prefix)
validation_gen = DataGenerator(val_json, n_classes=22, data_prefix=val_prefix)
```

- `df_json` and `val_json` are pandas dataframes containing the data.
- `n_classes` is the number of classes for the images.
- `data_prefix` is the prefix path for the image and text data.

3. Accessing data and labels:

The `DataGenerator` object provides the following methods to access the data and labels:

- `__getitem__(self, index)`: Returns a batch of data and labels for the given index.
- `__len__(self)`: Returns the number of batches per epoch.

Example usage:

```python
# Checking the shape of the tensors in the data loader
print(training_gen.__len__())

count = 0
for x, y in training_gen.__iter__():
    count += 1
    print("Shape of x:", len(x))
    print("Shape of tensor:", x[0].shape)
    print("Shape of y:", len(y))
    print("---\n")
    if count == 5:
        break

print(validation_gen.__len__())

count = 0
for x, y in validation_gen.__iter__():
    count += 1
    print("Shape of x:", len(x))
    print("Shape of tensor:", x[0].shape)
    print("Shape of y:", len(y))
    print("---\n")
    if count == 5:
        break
```

## Class Methods

The `DataGenerator` class provides the following methods:

- `__init__(self, df, batch_size=32, dim=(224, 224, 3), n_classes=22, data_prefix='', shuffle=True, scale=(224, 224))`: Initializes the `DataGenerator` object.
- `checkLabelSize(self, listOfLabels)`: Checks if any labels are missing and returns a list of missing label indices.
- `encodeData(self)`: Binarizes the data labels into 0's and 1's, adding missing labels if necessary.
- `on_epoch_end(self)`: Updates indexes after each epoch.
- `__data_generation(self, data_ids_temp, label_temp)`: Generates data containing batch_size samples.
- `__read_data_instance_image(self, pid, columnName)`: Returns the image data by reading the correct row from the dataframe.
- `__read_data_instance_text(self, pid, columnName)`: Returns the text data by reading the correct row from the dataframe.

## Note

This script relies on the following dependencies:

- `keras`: The Keras library for deep learning.
- `pandas`: The pandas library for data manipulation and analysis.
- `numpy`: The NumPy library for mathematical operations.
- `sklearn.preprocessing.MultiLabelBinarizer`: The scikit-learn library for binarizing multi-label data.
- `PIL.Image`: The Python Imaging Library for image processing.

Make sure to install these dependencies before running the script.

### Model Architecture

#### Baseline Models

##### Model with Resnet-50 and Bert En Cased L-24 H-1024 A-16 with no dense layer before the classifier

The model architecture consists of two branches: the image branch and the text branch.

**Image Branch**:
- Loads a pre-trained ResNet-50 model for meme image classification with weights pre-trained on ImageNet.
- Excludes the classifier layer at the top.
- The input image is preprocessed using the ResNet-50 preprocessing function.
- The processed image is passed through the ResNet-50 base model.
- The output is globally average pooled to obtain a fixed-size representation.

**Text Branch**:
- The text data in the dataset is in English, so the model uses a pre-trained BERT model that was trained on an English text corpus.
- The input text is preprocessed using the BERT preprocessing layer.
- The preprocessed text is encoded using the BERT model.
- The pooled output from BERT is obtained.

**Concatenation and Classification**:
- The outputs from the image branch and the text branch are concatenated.
- A sigmoid-activated dense layer with 22 units is added as the classifier.
- The final model takes both image and text inputs and outputs the classification probabilities.

##### Model with Resnet-50 and Bert En Cased L-24 H-1024 A-16 with one dense layer before the classifier

This model has a similar architecture to the previous baseline model, but with an additional dense layer before the classifier.

##### Model with Resnet-50 and Expert Bert(Trained on Wiki) with no dense layer before the classifier

This model uses an expert BERT model that was trained on a Wiki dataset instead of the previous BERT model. The rest of the architecture remains the same.

##### Model with Resnet-50 and Expert Bert(Trained on Wiki) with one dense layer before the classifier

This model combines ResNet-50 with the expert BERT model, similar to the previous model, but with an additional dense layer before the classifier.

### Model Training

Each model is compiled with the Adam optimizer and binary cross-entropy loss function. The F1 score is used as an additional metric. Checkpoint callbacks are set up to save the best model weights during training.

The models are trained using the `fit` method with 20 epochs and validation data. The training progress is logged, and learning curves are plotted for loss and F1 score.

### Usage

To use this script, make sure you have the required dependencies installed. You can modify the model architecture and training parameters according to your needs. Run the script and observe the training progress and performance of each model.

Note: The script assumes the availability of training and validation data generators (`training_gen` and `validation_gen`) and a helper function `get_callbacks()` to define the callbacks for model training.

**Disclaimer**: This script provides an overview of the model architecture and training process. Make sure to adapt it to your specific dataset and requirements before using it in a production environment.

Note: This script assumes it is running in a Google Colab environment and requires access to specific data files. Please adjust the file paths and configurations accordingly for your specific setup.

Feel free to modify the script or add additional functionality as needed for your specific image classification task.

## Model Architecture

### Baseline Models

#### Model with Resnet-50 and Bert En Cased L-24 H-1024 A-16 with no dense layer before the classifier

The model architecture consists of two branches: the image branch and the text branch. The image branch uses a pre-trained ResNet-50 model for image classification. The text branch utilizes a pre-trained BERT model trained on an English text corpus. The outputs of both branches are concatenated and passed through a sigmoid activation function to obtain the final classification. The model is trained using binary cross-entropy loss and the F1 score as the evaluation metric.

#### Model with Resnet-50 and Bert En Cased L-24 H-1024 A-16 with one dense layer before the classifier

This model is similar to the previous one, but it includes an additional dense layer before the final classifier. The dense layer helps to capture more complex patterns and relationships between the image and text features.

#### Model with Resnet-50 and Expert Bert (Trained on Wiki) with no dense layer before the classifier

This model combines a ResNet-50 model and an expert BERT model trained on a Wikipedia text corpus. The architecture is similar to the previous models, with separate image and text branches. The outputs are concatenated and passed through a sigmoid activation function to obtain the final classification.

#### Model with Resnet-50 and Expert Bert (Trained on Wiki) with one dense layer before the classifier

This model is an extension of the previous model with an additional dense layer before the final classifier. The dense layer adds more non-linearity to the model and can capture more complex relationships between the image and text features.

#### Model with EfficientNet B7 and Expert Bert (Trained on Wiki) with no dense layer before the classifier

This model uses an EfficientNet B7 model for image processing and an expert BERT model trained on a Wikipedia text corpus for text processing. The architecture follows the same pattern as the previous models, with separate image and text branches. The outputs are concatenated and passed through a sigmoid activation function for classification.

## Usage

1. Each model's architecture is defined and plotted using `tf.keras.utils.plot_model`. The summary of each model is also displayed.
2. The models are compiled with the Adam optimizer, binary cross-entropy loss, and the F1 score as the evaluation metric.
3. Training is performed on the provided training dataset for 20 epochs, with validation on a separate validation dataset. Model checkpoints are saved using `tf.keras.callbacks.ModelCheckpoint`.
4. Model performance is evaluated using custom functions to plot learning curves, displaying the loss and F1 score over epochs for both training and validation datasets.

## Results

For each model, the script displays learning curves showing the performance of the model on the entire dataset, including both training and validation datasets. The learning curves provide insights into the model's convergence, overfitting, and generalization capabilities.

## Model 1: EfficientNetB7 and Expert Bert (Trained on wiki) with no dense layer before the classifier

### Model Architecture
- Image Branch (CNN): EfficientNetB7 with pre-trained weights from ImageNet
- Text Branch (Bert): Expert Bert architecture trained on wiki books dataset
- Concatenation of the image and text features
- Classifier: Dense layer with sigmoid activation

### Training and Evaluation
- The model is compiled with the Adam optimizer and binary cross-entropy loss.
- F1-score is used as an additional evaluation metric.
- Model performance is saved and plotted using the `plot_learning_curve` function.

## Model 2: EfficientNetB7 and Expert Bert with one dense layer before the classifier

### Model Architecture
- Same as Model 1, but with an additional dense layer before the classifier.

### Training and Evaluation
- Same as Model 1.

## Model 3: EfficientNetB7 and Bert En Cased L-24 H-1024 A-16 with no dense layer before the classifier

### Model Architecture
- Image Branch (CNN): EfficientNetB7 with pre-trained weights from ImageNet
- Text Branch (Bert): Bert En Cased L-24 H-1024 A-16 architecture
- Concatenation of the image and text features
- Classifier: Dense layer with sigmoid activation

### Training and Evaluation
- Same as Model 1.

## Model 4: EfficientNetB7 and Bert En Cased L-24 H-1024 A-16 with one dense layer before the classifier

### Model Architecture
- Same as Model 3, but with an additional dense layer before the classifier.

### Training and Evaluation
- Same as Model 1.

## Hyperparameter Tuning
### Choice of models to be tuned:

Each baseline model was picked based on its performance, comparing the one with a dense layer before the classifier and the one without it. The model with ResNet-50 and Bert En Cased L-24 H-1024 A-16 without a dense layer was selected as its graph was less noisy compared to the one with the dense layer.

## 1. Tuning ResNet-50 and Bert En Cased L-24 H-1024 A-16 with no dense layer before the classifier

The first model being tuned is a combination of ResNet-50 and Bert En Cased L-24 H-1024 A-16 without a dense layer before the classifier. The hyperparameters being tuned include the learning rate and the dropout rate.

### Setting the tuner params

The hyperparameters learning rate and dropout rate are defined using the Keras Tuner API. The learning rate is chosen from three values: 5e-5, 3e-5, and 2e-5. The dropout rate is chosen from the range of 0 to 0.6 with a step of 0.1.

### Running Bayesian Optimization

The Bayesian Optimization tuner is used to search for the best hyperparameters for the model. The tuner is set to run a maximum of 3 trials. The search is performed using the training data and validated using the validation data.

### Searching for the best params

After the search is completed, the best hyperparameters found by the tuner are displayed.

## 2. Tuning ResNet-50 and Expert Bert with one dense layer before the classifier

The second model being tuned is a combination of ResNet-50 and Expert Bert with one dense layer before the classifier. Similar to the previous model, the learning rate and dropout rate are tuned using Bayesian Optimization.

## 3. Tuning EfficientNet B7 and Expert Bert with no dense layer before the classifier

The third model being tuned is a combination of EfficientNet B7 and Expert Bert without a dense layer before the classifier. Again, the learning rate and dropout rate are tuned using Bayesian Optimization.

## 4. Tuning EfficientNet B7 and Bert En Cased L-24 H-1024 A-16 with one dense layer before the classifier

The final model being tuned is a combination of EfficientNet B7 and Bert En Cased L-24 H-1024 A-16 with one dense layer before the classifier. The learning rate, dropout rate, and the number of units in the dense layer are tuned using Bayesian Optimization.

### Fitting the tuned models

After tuning the models, they can be fit to the data using the best hyperparameters obtained from the tuning process.

# Final Results and Model Selection

This section presents the final results and the selection of the best model based on the evaluation metrics. 

## Learning Curve

First, let's visualize the learning curves for the different models:

```python
plot_learning_curve(final_tuned_model_resnet_bert_result.history['loss'], final_tuned_model_resnet_bert_result.history['val_loss'],
                   final_tuned_model_resnet_bert_result.history['f1_score'], final_tuned_model_resnet_bert_result.history['val_f1_score'],
                    metric_name='f1')

plot_learning_curve(final_tuned_model_resnet_expert_dense.history['loss'], final_tuned_model_resnet_expert_dense.history['val_loss'],
                   final_tuned_model_resnet_expert_dense.history['f1_score'], final_tuned_model_resnet_expert_dense.history['val_f1_score'],
                    metric_name='f1')

plot_learning_curve(final_tuned_model_efficientNet_expert.history['loss'], final_tuned_model_efficientNet_expert.history['val_loss'],
                   final_tuned_model_efficientNet_expert.history['f1_score'], final_tuned_model_efficientNet_expert.history['val_f1_score'],
                    metric_name='f1')

plot_learning_curve(final_tuned_model_efficientNet_bert_dense.history['loss'], final_tuned_model_efficientNet_bert_dense.history['val_loss'],
                   final_tuned_model_efficientNet_bert_dense.history['f1_score'], final_tuned_model_efficientNet_bert_dense.history['val_f1_score'],
                    metric_name='f1')
```

## Selection of the Final Model

Based on the learning curves and the evaluation metrics, we have selected the final model. The chosen model is the EfficientNet and Expert BERT model with one dense layer before the classifier. This model showed the best performance on the validation set and achieved the highest F1 score.

## Preparing the Test Set

Next, we need to prepare the test set for evaluation. The test set is stored in the file `test_set_task3.txt`. We load the test set and encode the labels using the `encodeLabels` function.

```python
!cp /content/drive/'My Drive'/'Colab Notebooks'/'A2_DL'/'Subtask3'/test_set_task3.zip .
!unzip -q -o test_set_task3.zip
!rm test_set_task3.zip

df_test = pd.read_json('./test_set_task3/test_set_task3.txt')
df_test = encodeLabels(df_test)
```

## Test Set Statistics

Let's analyze some statistics of the test set to gain a better understanding:

```python
# Store all labels in a single list
allLabels=[]

for labels in df_test['labels']:
  for label in labels:
    allLabels.append(label)

allLabelsDf = pd.DataFrame(allLabels)
allLabelsDf.value_counts()

# Plotting the frequency of each label
sns.histplot(data=allLabelsDf)
```

## Predictions and Post Processing

Now, we will make predictions on the test set using the selected final model. After obtaining the predictions, we will apply post-processing to convert the float predictions to integer labels.

```python
predictions = final_model_efficientNet_dense_bert.predict(test_gen)

rounded_predictions = predictions.round()

# Convert the rounded predictions from float to int

finalPrediction=[]
for labels in rounded_predictions:
  newLabel=[]
  for label in labels:
    newLabel.append(int(label))
  finalPrediction.append(newLabel)

# Converting all of the 1's in the predictions to their respective encoded label
encodedPredictions = []
preds = []
for currentList in finalPrediction:
  temp = []
  i =

 0
  for label in currentList:
    if label == 1:
      temp.append(i)
      preds.append(i)
    i += 1
  encodedPredictions.append(temp)

sns.histplot(preds)
```

## True Labels

To evaluate the predictions, we need the true labels of the test set. We extract the true labels from the DataFrame `df_test`.

```python
correct_labels = pd.DataFrame(df_test['labels'])
```

## Adding Missing Labels

In some cases, the test set may not contain all possible labels. We need to ensure that the binarized label DataFrame includes all possible labels. The function `add_missing_labels` checks for missing labels and adds them as columns with default values of 0.

```python
def add_missing_labels(df):
    allowedLabels = [i for i in range(0,22)]
    missingLabels = []
    columns = df.columns

    # Checking which of the labels are missing
    for column in allowedLabels:
        if column not in columns:
            missingLabels.append(column)

    # Adding the missing labels to the binarized label dataframe and returning it
    for missingLabel in missingLabels:
        df.insert(loc = missingLabel, column = missingLabel, value = 0)

    return df

true_labels = df_test['labels'].tolist()

# Binarizing predictions
mlb = MultiLabelBinarizer()
mlb_labels_pred = mlb.fit_transform(encodedPredictions)
mlb_labels_pred = pd.DataFrame(mlb_labels_pred, columns = mlb.classes_ , dtype = 'int64')
mlb_labels_pred = add_missing_labels(mlb_labels_pred)

mlb_true_labels = mlb.fit_transform(true_labels)
mlb_true_labels = pd.DataFrame(mlb_true_labels, columns=mlb.classes_ , dtype='int64')
mlb_true_labels = add_missing_labels(mlb_true_labels)

y_pred = mlb_labels_pred
y_true = mlb_true_labels

print(classification_report(y_true, y_pred))
```
Feel free to modify the script according to your dataset and requirements. You can experiment with different architectures, hyperparameters, and training configurations to achieve better results.
