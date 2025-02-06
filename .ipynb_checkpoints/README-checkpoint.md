# Diabetes Prediction using Linear Regression

This markdown file explains the process of training a linear regression model to predict diabetes progression using the Body Mass Index (BMI) feature.

## Importing Necessary Libraries

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split 
```

We import essential libraries for data manipulation, visualization, and machine learning.

## Loading the Dataset

```python
diabetes = datasets.load_diabetes()
```

We use the `load_diabetes()` function from `sklearn.datasets` to load the diabetes dataset.

## Data Preprocessing

```python
# Selecting only the 3rd feature (Body Mass Index, BMI) for modeling.
diabetes_X = diabetes.data[:, np.newaxis, 2]
diabetes_y = diabetes.target
```

The dataset contains multiple features, but we select only the BMI feature for simplicity.

## Splitting Data into Training and Testing Sets

```python
diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = train_test_split(diabetes_X, diabetes_y, test_size=0.2, random_state=42)
```

We split the data into 80% training and 20% testing sets using `train_test_split()`.

## Training the Model

```python
model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_y_train)
```

A linear regression model is initialized and trained using the training dataset.

## Making Predictions

```python
diabetes_y_pred = model.predict(diabetes_X_test)
```

The model makes predictions on the test dataset.

## Evaluating Model Performance

```python
print("Mean square error: ", mean_squared_error(diabetes_y_pred, diabetes_y_test))
print("Coefficient: ", model.coef_)
print("Intercept: ", model.intercept_)
```

We compute the mean squared error, coefficient, and intercept to assess model performance.

### Sample Output:
```
Mean square error:  4061.8259284949263
Coefficient:  [998.57768914]
Intercept:  152.00335421448167
```

## Visualizing the Results

```python
plt.scatter(diabetes_X_test, diabetes_y_test)
plt.plot(diabetes_X_test, diabetes_y_pred)
plt.show()
```

We plot the actual vs predicted values to visualize model performance.

## Conclusion

This linear regression model predicts diabetes progression using BMI. The mean squared error provides insight into model accuracy, while visualization helps in understanding predictions.
