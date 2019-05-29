
# KNN with sklearn - Lab

## Introduction

In this lab, you'll learn how to use sci-kit learn's implementation of a KNN classifier on the classic Titanic dataset from Kaggle!

## Objectives

You will be able to:

* Use KNN to make classification predictions on a real-world dataset
* Perform a parameter search for 'k' to optimize model performance
* Evaluate model performance and interpret results

## Getting Started

Start by importing the dataset, stored in the `titanic.csv` file, and previewing it.


```python
import pandas as pd

raw_df = pd.read_csv('titanic.csv')
raw_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Great!  Next, you'll perform some preprocessing steps such as removing unnecessary columns and normalizing features.

## Preprocessing the Data

Preprocessing is an essential component to any data science pipeline. It's not always the most glamorous task as might be an engaging data visual or impressive neural network, but cleaning and normalizing raw datasets are required to produce useful and insightful datasets that form the backbone of all data powered projects. This can include changing column types, as in 
```python
df.col = df.col.astype('int')
```
or extracting subsets of information, such as 
```python
import re
df.street = df.address.map(lambda x: re.findall('(.*)?\n', x)[0])
```

> **Note:** While outside the scope of this particular lesson, **regular expressions** (mentioned above) are powerful tools for pattern matching!  
See the [regular expressions official documentation here](https://docs.python.org/3.6/library/re.html). 

Since you've done this before, you should be able to do this quite well yourself without much hand holding by now. 

In the cells below, complete the following steps:

1. Remove unnecessary columns (PassengerId, Name, Ticket, and Cabin).
2. Convert `Sex` to a binary encoding, where female is `0` and male is `1`.
3. Detect and deal with any null values in the dataset. 
    * For `Age`, replace null values with the median age for the dataset. 
    * For `Embarked`, drop the rows that contain null values
4. One-Hot Encode categorical columns such as `Embarked`.
5. Store the target column, `Survived`, in a separate variable and remove it from the DataFrame. 


```python
df = raw_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=False)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.Sex = df.Sex.map({'female': 0, 'male': 1})
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isna().sum()
```




    Survived      0
    Pclass        0
    Sex           0
    Age         177
    SibSp         0
    Parch         0
    Fare          0
    Embarked      2
    dtype: int64




```python
df.Age = df.Age.fillna(df.Age.median())
df.isna().sum()
```




    Survived    0
    Pclass      0
    Sex         0
    Age         0
    SibSp       0
    Parch       0
    Fare        0
    Embarked    2
    dtype: int64




```python
df = df.dropna()
df.isna().sum()
```




    Survived    0
    Pclass      0
    Sex         0
    Age         0
    SibSp       0
    Parch       0
    Fare        0
    Embarked    0
    dtype: int64




```python
one_hot_df = pd.get_dummies(df)
one_hot_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
labels = one_hot_df.Survived
one_hot_df.drop('Survived', axis=1, inplace=True)
```

## Normalizing the Data

The final step in your preprocessing efforts for this lab is to **_normalize_** the data. Remember that normalization (also sometimes called **_Standardization_** or **_Scaling_**) means making sure that all of your data is represented at the same scale.  The most common way to do this is to convert all numerical values to z-scores. 

Since KNN is a distance-based classifier, if data is different scales, then larger scaled features would have a larger impact on the distance between points.

To scale your data, use the `StandardScaler` object found inside the `sklearn.preprocessing` module. 

In the cell below:

* Import and instantiate a `StandardScaler` object. 
* Use the scaler's `.fit_transform()` method to create a scaled version of our dataset. 
* The result returned by the `fit_transform` call will be a numpy array, not a pandas DataFrame. Create a new pandas DataFrame out of this object called `scaled_df`. To set the column names back to their original state, set the `columns` parameter to `one_hot_df.columns`.
* Print out the head of `scaled_df` to ensure everything worked correctly.


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(one_hot_df)

scaled_df = pd.DataFrame(scaled_data, columns=one_hot_df.columns)
scaled_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.825209</td>
      <td>0.735342</td>
      <td>-0.563674</td>
      <td>0.431350</td>
      <td>-0.474326</td>
      <td>-0.500240</td>
      <td>-0.482711</td>
      <td>-0.307941</td>
      <td>0.616794</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.572211</td>
      <td>-1.359911</td>
      <td>0.669217</td>
      <td>0.431350</td>
      <td>-0.474326</td>
      <td>0.788947</td>
      <td>2.071634</td>
      <td>-0.307941</td>
      <td>-1.621287</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.825209</td>
      <td>-1.359911</td>
      <td>-0.255451</td>
      <td>-0.475199</td>
      <td>-0.474326</td>
      <td>-0.486650</td>
      <td>-0.482711</td>
      <td>-0.307941</td>
      <td>0.616794</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.572211</td>
      <td>-1.359911</td>
      <td>0.438050</td>
      <td>0.431350</td>
      <td>-0.474326</td>
      <td>0.422861</td>
      <td>-0.482711</td>
      <td>-0.307941</td>
      <td>0.616794</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.825209</td>
      <td>0.735342</td>
      <td>0.438050</td>
      <td>-0.475199</td>
      <td>-0.474326</td>
      <td>-0.484133</td>
      <td>-0.482711</td>
      <td>-0.307941</td>
      <td>0.616794</td>
    </tr>
  </tbody>
</table>
</div>



You may have noticed that the scaler also scaled our binary/one-hot encoded columns, too! Although it doesn't look as pretty, this has no negative effect on the model. Each 1 and 0 have been replaced with corresponding decimal values, but each binary column still only contains 2 values, meaning the overall information content of each column has not changed. 

#### Creating Training and Testing Sets

Now that you've preprocessed the data, the only step remaining is to split our data into training and testing sets. 

In the cell below:

* Import `train_test_split` from the `sklearn.model_selection` module
* Use `train_test_split` to split thr data into training and testing sets, with a `test_size` of `0.25`.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(one_hot_df, labels, test_size=0.25)
```

## Fitting a KNN Model

Now that you've preprocessed the data it's time to train a KNN classifier and validate its accuracy. 

In the cells below:

* Import `KNeighborsClassifier` from the `sklearn.neighbors` module.
* Instantiate a classifier. For now, you can just use the default parameters. 
* Fit the classifier to the training data/labels
* Use the classifier to generate predictions on the test data. Store these predictions inside the variable `test_preds`.


```python
from sklearn.neighbors import KNeighborsClassifier
clf1 = KNeighborsClassifier()
clf1.fit(X_train, y_train)
test_preds = clf1.predict(X_test)
```

Now, in the cells below, import all the necessary evaluation metrics from `sklearn.metrics` and complete the `print_metrics()` function so that it prints out **_Precision, Recall, Accuracy,_** and **_F1-Score_** when given a set of `labels` (the true values) and `preds` (the models predictions). 

Finally, use `print_metrics()` to print out the evaluation metrics for the test predictions stored in `test_preds`, and the corresponding labels in `y_test`.


```python
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
```


```python
def print_metrics(labels, preds):
    print("Precision Score: {}".format(precision_score(labels, preds)))
    print("Recall Score: {}".format(recall_score(labels, preds)))
    print("Accuracy Score: {}".format(accuracy_score(labels, preds)))
    print("F1 Score: {}".format(f1_score(labels, preds)))
    
print_metrics(y_test, test_preds)
```

    Precision Score: 0.5934065934065934
    Recall Score: 0.6136363636363636
    Accuracy Score: 0.6816143497757847
    F1 Score: 0.6033519553072626


> **_Analysis_** Interpret each of the metrics above, and explain what they tell you about your model's capabilities. If you had to pick one score to best describe the performance of the model, which would you choose? Explain your answer.

Write your answer below this line:
________________________________________________________________________________



## Improving Model Performance

While your overall model results should be better than random chance, they're probably mediocre at best given that you haven't tuned the model yet. For the remainder of this notebook, you'll focus on improving your model's performance. Remember that modeling is an **_iterative process_**, and developing a baseline out of the box model such as the one above is always a good start. 

First, try to find the optimal number of neighbors to use for the classifier. To do this, complete the `find_best_k()` function below to iterate over multiple values of k and find the value of k that returns the best overall performance. 

The skeleton function takes in six parameters:
* `X_train`
* `y_train`
* `X_test`
* `y_test`
* `min_k` (default is 1)
* `max_k` (default is 25)
    
> **Pseudocode Hint**:
1. Create two variables, `best_k` and `best_score`
1. Iterate through every **_odd number_** between `min_k` and `max_k + 1`. 
    1. For each iteration:
        1. Create a new KNN classifier, and set the `n_neighbors` parameter to the current value for k, as determined by our loop.
        1. Fit this classifier to the training data.
        1. Generate predictions for `X_test` using the fitted classifier.
        1. Calculate the **_F1-score_** for these predictions.
        1. Compare this F1-score to `best_score`. If better, update `best_score` and `best_k`.
1. Once all iterations are complete, print out the best value for k and the F1-score it achieved.


```python
def find_best_k(X_train, y_train, X_test, y_test, min_k=1, max_k=25):
    best_k = 0
    best_score = 0.0
    for k in range(min_k, max_k+1, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        f1 = f1_score(y_test, preds)
        if f1 > best_score:
            best_k = k
            best_score = f1
    
    print("Best Value for k: {}".format(best_k))
    print("F1-Score: {}".format(best_score))

```


```python
find_best_k(X_train, y_train, X_test, y_test)
# Expected Output:

# Best Value for k: 3
# F1-Score: 0.6444444444444444
```

    Best Value for k: 3
    F1-Score: 0.6444444444444444


If all went well, you'll notice that model performance has improved by over 4 percent by finding an optimal value for k. For further tuning, you can use sci-kit learn's built in **Grid Search** to perform a similar exhaustive check of hyper-parameter combinations and fine tune model performance. For a full list of model parameters, see the [sklearn documentation !](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

## (Optional) Level Up: Iterating on the Data

As an optional (but recommended!) exercise, think about the decisions you made during the preprocessing steps that could have affected the overall model performance. For instance, you were asked to replace the missing age values with the column median. Could this have affected the overall performance? How might the model have fared if you had just dropped those rows, instead of using the column median? What if you reduced the data's dimensionality by ignoring some less important columns altogether?

In the cells below, revisit your preprocessing stage and see if you can improve the overall results of the classifier by doing things differently.Consider dropping certain columns, dealing with null values differently, or using an alternative scaling function. Then see how these different preprocessing techniques affect the performance of the model. Remember that the `find_best_k` function handles all of the fitting&mdash;use this to iterate quickly as you try different strategies for dealing with data preprocessing! 



```python

```


```python

```


```python

```


```python

```


```python

```

## Summary

Well done! In this lab, you worked with the classic titanic dataset and practiced fitting and tuning KNN classification models using sci-kit learn! As always, this gave you another opportunity to continue practicing your data wrangling skills and model tuning skills using pandas and sci-kit learn!
