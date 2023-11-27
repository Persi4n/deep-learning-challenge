# AlphabetSoupCharity Neural Network Model

## Overview

This project focuses on creating a binary classification model using a neural network to predict the success of organizations funded by Alphabet Soup. The model utilizes various features from the provided dataset to determine the effectiveness of funding allocation.

## Project Structure

The project is organized into the following sections:

1. **Preprocessing**: 
   - Removal of non-beneficial columns.
   - Binning of rare values in 'APPLICATION_TYPE' and 'CLASSIFICATION'.
   - Conversion of categorical data to numeric using `pd.get_dummies`.
   - Splitting data into features and target arrays.
   - Scaling of data using StandardScaler.

2. **Model Building and Training**:
   - Definition of a deep neural network with appropriate architecture.
   - Compilation of the model using Adam optimizer and binary crossentropy loss.
   - Training the model on the training dataset for 100 epochs.

3. **Model Evaluation**: 
   - Evaluation of the model on the test dataset, obtaining accuracy and loss metrics.

4. **Model Optimization**: 
   - Exploration of optimization methods, including adjusting input data and modifying neural network architecture.

5. **Model Export**: The trained model is exported to an HDF5 file named "AlphabetSoupCharity.h5".


## Dependencies

- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [tensorflow](https://www.tensorflow.org/)

## Instructions

1. Clone the repository: `git clone`
2. Navigate to the project directory: `cd AlphabetSoupCharity`
3. Open and run the Jupyter Notebooks for preprocessing, model building, and optimization.
4. Examine the model results in the evaluation section of the Jupyter Notebook.
5. Exported model can be found in the file `AlphabetSoupCharity.h5`.


# Neural Network Model Analysis for AlphabetSoupCharity

## Introduction

The purpose of this analysis is to design, train, and evaluate a neural network model for Alphabet Soup's funding allocation. The goal is to create a binary classification model that predicts the success of an organization based on various features provided in the dataset.

## Model Architecture and Training

### Model Structure
We designed a deep neural network with an input layer, one or more hidden layers, and an output layer. The number of nodes in each layer and the activation functions were chosen based on considerations of the dataset.

```python
# Model Definition
nn = tf.keras.models.Sequential()
nn.add(tf.keras.layers.Dense(units=10, input_dim=len(X_train.columns), activation='relu'))
nn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
```

### Training
The model was compiled using the Adam optimizer, binary crossentropy loss, and accuracy as the metric. It was then trained on the preprocessed data with 100 epochs.

```python
# Model Compilation
nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model Training
nn.fit(X_train_scaled, y_train, epochs=100, validation_data=(X_test_scaled, y_test))
```

## Model Evaluation

### Results
After training, the model was evaluated on the test dataset, yielding the following results:

```python
# Model Evaluation
model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```

### Questions

1. **What variable(s) are the target(s) for your model?**
   - The target variable for our model is 'IS_SUCCESSFUL.'

2. **What variable(s) are the features for your model?**
   - The features for our model include various columns such as 'APPLICATION_TYPE,' 'AFFILIATION,' 'CLASSIFICATION,' and others.

3. **What variable(s) should be removed from the input data because they are neither targets nor features?**
   - The 'EIN' and 'NAME' columns were removed as they do not contribute to the target or features.

4. **How many neurons, layers, and activation functions did you select for your neural network model, and why?**
   - We selected 10 neurons for the hidden layer with a 'relu' activation function. This choice was made based on experimentation and considerations for the dataset.

5. **Were you able to achieve the target model performance?**
   - The achieved model performance can be seen in the accuracy and loss values obtained during evaluation.

6. **What steps did you take in your attempts to increase model performance?**
   - Possible steps could include adjusting the number of neurons, layers, activation functions, or exploring different optimization algorithms.

## Model Summary

The model demonstrated promising results with an accuracy of [accuracy_value] and a loss of [loss_value]. Further optimization and fine-tuning could potentially enhance performance.

## Alternative Model Recommendation

For a different approach to solving this classification problem, a Random Forest or Gradient Boosting model could be considered. These ensemble methods often perform well on structured datasets and can handle non-linear relationships effectively. Additionally, they provide feature importance insights, which may aid in interpreting the model's decisions.
