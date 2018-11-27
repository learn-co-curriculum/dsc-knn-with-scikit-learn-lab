
# KNN with sklearn - Lab

## Introduction

In this lab, we'll learn how to use sklearn's implementation of a KNN classifier  on some real world datasets!

## Objectives

You will be able to:

* Use KNN to make classification predictions on a real-world dataset
* Perform a parameter search for 'k' to optimize model performance
* Evaluate model performance and interpret results

## Getting Started

In this lab, we'll make use of sklearn's implementation of the **_K-Nearest Neighbors_** algorithm. We'll use it to make predictions on the Titanic dataset. 

We'll start by importing the dataset, and then deal with preprocessing steps such as removing unnecessary columns and normalizing our dataset.

You'll find the titanic dataset stored in the `titanic.csv` file. In the cell below:

* Import pandas and set the standard alias.
* Read in the data from `titanic.csv` and store it in a pandas DataFrame. 
* Print the head of the DataFrame to ensure everything loaded correctly.

Great! Now, we'll preprocess our data to get it ready for use with a KNN classifier.

## Preprocessing Our Data

This stage should be pretty familiar to you by now. Although it's not the fun part of machine learning, it's good practice to get used to it.  Although it isn't as fun or exciting as training machine learning algorithms, it's a very large, very important part of the Data Science Process. As a Data Scientist, you'll often spend the majority of your time wrangling and preprocessing, just to get it ready for use with supervised learning algorithms. 

Since you've done this before, you should be able to do this quite well yourself without much hand holding by now. 

In the cells below, complete the following steps:

1. Remove unnecessary columns (PassengerId, Name, Ticket, and Cabin).
2. Convert `Sex` to a binary encoding, where female is `0` and male is `1`.
3. Detect and deal with any null values in the dataset. 
    * For `Age`, replace null values with the median age for the dataset. 
    * For `Embarked`, drop the rows that contain null values
4. One-Hot Encode categorical columns such as `Embarked`.
5. Store our target column, `Survived`, in a separate variable and remove it from the DataFrame. 

## Normalizing Our Data

Good job preprocessing our data! This can seem tedious, but its a very important foundational skill in any Data Science toolbox. The final step we we'll take in our preprocessing efforts is to **_Normalize_** our data. Recall that normalization (also sometimes called **_Standardization_** or **_Scaling_**) means making sure that all of our data is represented at the same scale.  The most common way to do this is to convert all numerical values to z-scores. 

Since KNN is a distance-based classifier, data on different scales and negatively affect the results of our model! Predictors on much larger scales will overwhelm data with much smaller scales, because euclidean distance is going to treat them as the same.

To scale our data, we'll make use of the `StandardScaler` object found inside the `sklearn.preprocessing` module. 

In the cell below:

* Import and instantiate a `StandardScaler` object. 
* Use the scaler's `.fit_transform()` method to create a scaled version of our dataset. 
* The result returned by the `fit_transform` call will be a numpy array, not a pandas DataFrame. Create a new pandas DataFrame out of this object called `scaled_df`. To set the column names back to their original state, set the `columns` parameter to `one_hot_df.columns`.
* Print out the head of `scaled_df` to ensure everything worked correctly.


```python
# Dont forget to import!

scaler = None
scaled_data = None

scaled_df = None
scaled_df.head()
```

You may have noticed that the scaler also scaled our binary/one-hot encoded columns, too! Although it doesn't look as pretty, this has no negative effect on our model. Each 1 and 0 have been replaced with corresponding decimal values, but each binary column still only contains 2 values, meaning the overall information content of each column has not changed. 

#### Creating Training and Testing Sets

Now that we've preprocessed our data, the only step remaining is to split our data into training and testing sets. 

In the cell below:

* Import `train_test_split` from the `sklearn.model_selection` module
* Use `train_test_split` to split our data into training and testing sets, with a `test_size` of `0.25`.

## Creating and Fitting our KNN Model

Now that we've preprocessed our data successfully, it's time for the fun stuff--let's create a KNN classifier and use it to make predictions on our dataset!  Since you've got some experience on this part from when we built our own model, we won't hold your hand through section. 

In the cells below:

* Import `KNeighborsClassifier` from the `sklearn.neighbors` module.
* Instantiate a classifier. For now, we'll just use the default parameters. 
* Fit the classifier to our training data/labels
* Use the classifier to generate predictions on our testing data. Store these predictions inside the variable `test_preds`.

Now, in the cells below, import all the necessary evaluation metrics from `sklearn.metrics` abd then complete the following `print_metrics()` function so that it prints out **_Precision, Recall, Accuracy,_** and **_F1-Score_** when given a set of `labels` and `preds`. 

Then, use it to print out the evaluation metrics for our test predictions stored in `test_preds`, and the corresponding labels in `y_test`.


```python
def print_metrics(labels, preds):
    print("Precision Score: {}".format(None))
    print("Recall Score: {}".format(None))
    print("Accuracy Score: {}".format(None))
    print("F1 Score: {}".format(None))
    
print_metrics(y_test, test_preds)
```

**_QUESTION:_** Interpret each of the metrics above, and explain what they tell us about our model's capabilities. If you had to pick one score to best describe the performance of the model, which would you choose? Explain your answer.

Write your answer below this line:
________________________________________________________________________________



## Improving Model Performance

Our overall model results are better than random chance, but not by a large margin. For the remainder of this notebook, we'll focus on improving model performance. This is also a big part of the Data Science Process--your first fit is almost never your best. Modeling is an **_iterative process_**, meaning that we should make small incremental changes to our model and use our intuition to see if we can improve the overall performance. 

First, we'll start off by trying to find the optimal number of neighbors to use for our classifier. To do this, we'll write a quick function that iterates over multiple values of k and finds the one that returns the best overall performance. 

In the cell below, complete the `find_best_k()` function.  This function should:

* take in six parameters:
    * `X_train`, `y_train`, `X_test`, and  `y_test`
    * `min_k` and `max_k`. Set these to `1` and `25`, by default
* Create two variables, `best_k` and `best_score`
* Iterate through every **_odd number_** between `min_k` and `max_k + 1`. 
* For each iteration:
    * Create a new KNN classifier, and set the `n_neighbors` parameter to the current value for k, as determined by our loop.
    * Fit this classifier to the training data.
    * Generate predictions for `X_test` using the fitted classifier.
    * Calculate the **_F1-score_** for these predictions.
    * Compare this F1-score to `best_score`. If better, update `best_score` and `best_k`.
* Once it has checked every value for `k`, print out the best value for k and the F1-score it achieved.


```python
def find_best_k(X_train, y_train, X_test, y_test, min_k=1, max_k=25):
    pass

```


```python
find_best_k(X_train, y_train, X_test, y_test)
# Expected Output:

# Best Value for k: 3
# F1-Score: 0.6444444444444444
```

We improved our model performance by over 4 percent just by finding an optimal value for k. Good job! There are other parameters in the model that you can also tune. In a later section, we'll cover how we can automate the parameter search process using a technique called **_Grid Search_**. For, try playing around with the different options for parameters, and seeing how it affects model performance. For a full list of model parameters, see the [sklearn documentation !](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

## (Optional) Level Up: Iterating on the Data

As an optional (but recommended!) exercise, think about the decisions we made during the preprocessing steps that could have affected our overall model performance. For instance, we replaced missing age values with the column median. Could this have affected ourn overall performance? How might the model have fared if we had just dropped those rows, instead of using the column median? What if we reduced dimensionality by ignoring some less important columns altogether?

In the cells below, revisit your preprocessing stage and see if you can improve the overall results of the classifier by doing things differently. Perhaps you should consider dropping certain columns, or dealing with null values differently, or even using a different sort of scaling (or none at all!). Try a few different iterations on the preprocessing and see how it affects the overall performance of the model. The `find_best_k` function handles all of the fitting--use this to iterate quickly as you try different strategies for dealing with data preprocessing! 


## Summary

Good job! This concludes today's section!
