# Machine Learning models for soccer prediction

* Author: Andrew To

## Project Description

This project predicts the outcomes of soccer matches (Home Win, Draw, Away Win) using machine learning models. Currently, I am using the data of La Liga's matches from 2008 to 2020. The goal of this project is to develop and evaluate multiple machine learning models to identify the most effective approach for predicting match outcomes. I have implemented and compared the performance of the following models:

* Decision Tree
* Random Forest
* Gradient Boosting
* NB Guassian
* XgBoost 
* CatBoost

## Data Summary

### Data source: 
- Sports-Statistics.com 
- Links: https://sports-statistics.com/sports-data/soccer-datasets/

### Data Description:

The dataset used in this project combines information about historical soccer match statistics and bettings odds. Below are the key components of the dataset:

#### Historical Match Statistics:
  * Match Results: Outcome of each match (Home Win/Draw/Away Win)
  * Goals Scored: Home Goals/Away Goals
  * Other Features: Home Shot/Away Shot; Home Shot On Target/Away Shot On Target; Corners; Yellow/Red Cards;...

#### Betting Odds:

  * Odds Data: Pre-match betting odds for match outcomes and Asian handicap odds provided by various betting companies

Features description are stored in the file ../data/description.yaml

#### Dataset Structure:

   * 50+ features and about 180 games for each seasons, with total of 11 seasons in training dataset and 1 season in testing dataset. Datasets in different years have different number of features in betting odds, but they have the same features for match statistics

#### Data preprocessing:

   * Data featuring: Since we can't use directly the historical outcomes as a feature in testing datasets (lead to overfitting), I combined multiple features to make a measurement for recent performances of each team. New features can be total points in recent 5 matches, total goals in recent 5 matches, total shot on target in recent 5 matches,...

   * Encoding: 

        **Label Encoding**: Assigns a unique integer to result of each match (0: Away Win; 1: Draw; 2: Home Win)
     
        **One-Hot Encoding**: Creates binary columns for each team, with a `1` indicating the presence of a particular team.

   * Data scaling: This transformation transform the raw betting odds to equivalent winning probabilities and guarantees that the sum of normalized probabilities of all outcomes 1 for each match. Steps and formulas:

        1. **Convert Odds to Probabilities**:  
            Each betting odd  $O_{i,j}$ is converted to a probability $P_{i,j}$ using:  
            <p align="center">$$\large P_{i,j} = \frac{1}{O_{i,j}}$$</p>


        2. **Compute Normalization Factor**:  
        The sum of probabilities for all outcomes in a match is calculated as:  
            <p align="center">$$\large \text{Normalization Factor}_{i} = \sum_{j=1}^{n} P_{i,j}$$</p>
            where $n$ is the total number of provided outcomes (Home Win, Draw, Away Win)
        

        3. **Normalize Probabilities**:  
        Each probability is divided by the normalization factor:
            <p align="center">$$\large P'_{i,j} = \frac{P_{i,j}}{\text{Normalization Factor}_{i}}$$</p>

## Models Training

### Data Preparation:
 * Training dataset: I combined dataset from 2008-2009 season to 2018-2019 season as trainning dataset
 * Testing dataset: The dataset of the 2019-2020 season

### Models:

Models that were used for training and testing:
 * Decision Tree
 * Random Forest
 * Gradient Boosting
 * NB Gaussian
 * XGBoost
 * CatBoost

### Evaluation Metrics:

Use classification report to generate the following evaluation metrics: 

 * Overall test accuracy
 * Precision
 * Recall
 * F1-score
 * Support

### Training process:

#### Hyperparameter Tuning:

 * I used GridSearchCV to systematically and automatically search for the best hyperparameters. The parameters are flexible and depend on the models used for training. The default scoring metric across models is balanced accuracy, which aims to achieve the best performance beyond simple accuracy.

## Results

### Decision Tree

#### Test Set Accuracy: 0.49

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.42      | 0.53   | 0.47     | 47      |
| 1     | 0.29      | 0.08   | 0.12     | 50      |
| 2     | 0.57      | 0.72   | 0.63     | 83      |

| Metric          | Precision | Recall | F1-Score |
|------------------|-----------|--------|----------|
| Accuracy         |           |        | 0.49     |
| Macro Avg        | 0.42      | 0.44   | 0.41     |
| Weighted Avg     | 0.45      | 0.49   | 0.45     |

### Random Forest

#### Test Set Accuracy: 0.52

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.42      | 0.36   | 0.39     | 47      |
| 1     | 0.48      | 0.24   | 0.32     | 50      |
| 2     | 0.57      | 0.78   | 0.66     | 83      |

| Metric          | Precision | Recall | F1-Score |
|------------------|-----------|--------|----------|
| Accuracy         |           |        | 0.52     |
| Macro Avg        | 0.49      | 0.46   | 0.46     |
| Weighted Avg     | 0.50      | 0.52   | 0.49     |

### Gradient Boosting

### Naive Bayes

#### Test Set Accuracy: 0.49

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.46      | 0.55   | 0.50     | 47      |
| 1     | 0.34      | 0.46   | 0.39     | 50      |
| 2     | 0.70      | 0.47   | 0.56     | 83      |

| Metric          | Precision | Recall | F1-Score |
|------------------|-----------|--------|----------|
| Accuracy         |           |        | 0.49     |
| Macro Avg        | 0.50      | 0.49   | 0.48     |
| Weighted Avg     | 0.54      | 0.49   | 0.50     |

### Stacking Classifier

Stacking classifiers: Gradient Boosting, Random Forest; final classifier: Naive Bayes

#### Test Accuracy: 0.53

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.46      | 0.62   | 0.53     | 47      |
| 1     | 0.42      | 0.30   | 0.35     | 50      |
| 2     | 0.64      | 0.63   | 0.63     | 83      |

| Metric          | Precision | Recall | F1-Score |
|------------------|-----------|--------|----------|
| Accuracy         |           |        | 0.53     |
| Macro Avg        | 0.51      | 0.51   | 0.50     |
| Weighted Avg     | 0.53      | 0.53   | 0.53     |

### CatBoost

#### Test Accuracy: 0.56

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.51      | 0.47   | 0.49     | 47      |
| 1     | 0.49      | 0.40   | 0.44     | 50      |
| 2     | 0.61      | 0.71   | 0.66     | 83      |

| Metric          | Precision | Recall | F1-Score |
|------------------|-----------|--------|----------|
| Accuracy         |           |        | 0.56     |
| Macro Avg        | 0.54      | 0.53   | 0.53     |
| Weighted Avg     | 0.55      | 0.56   | 0.55     |




    

