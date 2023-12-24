# Body-Type-Classification-With-Logistic-Regression
The aim of this project is to process the Logistic Regression


## How to start with DOCKER
### Setup
+ ```docker container run --name classificationapi  -p 8081:8000 -d tahapek5454/body_type_classification:latest```
+ go ```http://127.0.0.1:8081/docs```

### Steps
+ First, you run fitModel
+ after that you can predict data

Note: SetData only work your local not in container

# Logistic Regression

Logistic Regression is a classification algorithm commonly used for binary classification problems. This algorithm attempts to separate data points into two different classes by using a linear combination of independent variables.

## Basic Principles

- **Objective**: To obtain a binary output (usually 0 and 1).
- **Model**: A linear function and a sigmoid function that transforms the result into a log-odds ratio.
- **Training**: The model learns its parameters (weights) by fitting to the training data.

## Sigmoid Function

The sigmoid (logistic) function used in Logistic Regression is defined as:

\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]

This function transforms any real number (z) into a value between 0 and 1.

## Model Equation

The fundamental equation for the Logistic Regression model is:

\[ P(Y=1|X) = \frac{1}{1 + e^{-(b + w_1x_1 + w_2x_2 + \ldots + w_nx_n)}} \]

In this equation:
- \( P(Y=1|X) \): Probability of being class 1 for a given input (X).
- \( b \): Bias or intercept term.
- \( w_1, w_2, \ldots, w_n \): Weights.
- \( x_1, x_2, \ldots, x_n \): Input features.

## Training and Prediction

1. **Training**: Model parameters are trained on the dataset.
2. **Prediction**: The trained model makes classification predictions for new input features.

## Advantages and Disadvantages

### Advantages
- Simple and fast.
- Performs well, especially when there is a linear relationship between features.

### Disadvantages
- Cannot model non-linear relationships.
- Prone to overfitting, so proper regularization may be needed.
- The dependent variable must be categorical (binary classification).

## Application

Logistic Regression is widely used in various fields such as medicine, economics, marketing, and bioinformatics to solve classification problems.


