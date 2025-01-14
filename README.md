# Machine Learning models for soccer prediction

* Author: Andrew To

## Project Description

This project predicts the outcomes of soccer matches (Home Win, Draw, Away Win) using machine learning models. Currently, I am using the data of La Liga's matches from 2008 to 2020. The goal of this project is to develop and evaluate multiple machine learning models to identify the most effective approach for predicting match outcomes. I have implemented and compared the performance of the following models:

* Decision Tree
* Random Forest
* Gradient Boosting
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



    

