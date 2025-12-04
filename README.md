# Breast-Cancer-Prediction-Project
## Project title:
***Prediction of Breast Cancer Using Data Collected from Fine Needle Aspiration (FNA) method.***

## Purpose and Outcome:
  - Purpose: This study assesses the correlation between the collected features and the benign or malignant nature of breast tumors. Based on this analysis, a predictive model is developed to determine tumor malignancy.
  - Outcome: Identify features that are strongly associated with breast cancer and to use them to accurately predict the malignancy of tumors.
## Dataset:
### Description:
  The Breast Cancer Wisconsin (Diagnostic) Dataset is a medical dataset widely used to predict whether a breast tumor is malignant (M) or benign (B). The data is derived from digitized images of fine needle aspirates (FNA) of breast masses, and the features describe cell nuclei characteristics.
  - Total Samples: 569
  - Features: 30
  - Target variable: Diagnosis (M= Malignant, B = Benign)

### Source: Wisconsin Diagnostic Breast Cancer (WDBC) dataset https://www.kaggle.com/datasets/khansaafreen/breastdataset?select=data.cs

### Structure:
#### Features Explained: The dataset contains mean, standard error (SE), and worst values for each measurement of the tumor’s shape and texture
  
`Radius`: Radius of the tumor
  
`Texture`: Variation in grey-scale intensity
  
`Perimeter`: Tumor’s boundary length
  
`Area`: Size of the tumor 
  
`Smoothness`: How smooth the tumor surface is
  
`Compactness`: How compact or irregular the tumor is
  
`Concavity`: How deep the indentations are
  
`Concave Points`: Number of concave portions
  
`Symmetry`: Tumor symmetry
  
`Fractal Dimension`: “Roughness” of the tumor boundary
#### ID and Target column:
`ID`
  
`Diagnosis`: M= Malignant, B = Benign

## Process - Step on Google Colab

## 1. Load Data

### Import packages
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
```
### Import data
```python
df = pd.read_csv('/content/drive/MyDrive/Project/Breast_Cancer_Prediction_Data.csv')
```
## 2. Data Cleaning

### Remove unesscessary columns
Remove columns: `id`, `Unnamed: 32`, 
Remove standard error features:
```python
se_cols = df.columns[df.columns.str.contains('se', case=False)]
df = df.drop(columns = ['Unnamed: 32', 'id'])
df = df.drop(columns = se_cols)
```
### Check duplicate data
``python
df.duplicated().sum()
``
**No duplicate data on this dataset**
### Check and Handle missing values
```python
df.isna().sum()
```
**There is no missing value in the dataset**
### Check Target column
```python
df['diagnosis'].value_counts(normalize=True).mul(100).plot(kind='pie', autopct='%1.2f%%', figsize=(5, 5))

plt.ylabel('')
plt.title('% Diagnosis')
plt.show()
```
![% Diagnosis](https://github.com/ducnguyen8600/Breast-Cancer-Prediction-Project/blob/main/%25%20Diagnosis.png)

## 3. Exploratory Data Analysis

### Divide features into 2 groups include mean features and worst features
```python
mean_cols = df.columns[df.columns.str.contains('mean', case=False)].to_list()
worst_cols = df.columns[df.columns.str.contains('worst', case=False)].to_list()
```
### Dataset Central Tendency
We check Dataset Descriptive Statistic and try to remove outliers.
But **Deleting outliers results in a significant decrease in the number of 'M' values in the target column, so we decide not to remove outliers.**

### Mean features Central tendency
```python
fig, axes = plt.subplots(1, 10, figsize=(20,4))
fig.suptitle("Central Tendency of Mean Features", fontsize = 15)
for i, col in enumerate(mean_cols):
    axes[i].boxplot(df[col], patch_artist=True)
    axes[i].set_title(col)
plt.tight_layout()
plt.show()
```
![](https://github.com/ducnguyen8600/Breast-Cancer-Prediction-Project/blob/main/Mean%20Features%20Central%20Tendency.png)

#### Mean features central tendency by Diagnosis

**Benign Diagnosis**
```python
fig, axes = plt.subplots(1, 10, figsize=(20,4))
fig.suptitle("Central Tendency of Mean Features in Benign Diagnosis", fontsize = 15)
for i, col in enumerate(mean_cols):
    axes[i].boxplot(df[col][df['diagnosis'] == 'B'], patch_artist=True)
    axes[i].set_title(col)

plt.tight_layout()
plt.show()
```
![](https://github.com/ducnguyen8600/Breast-Cancer-Prediction-Project/blob/main/Mean%20Features%20Central%20tendency%20in%20Benign%20Diagnosis.png)

**Malignant Diagnosis**
```python
fig, axes = plt.subplots(1, 10, figsize=(20,4))
fig.suptitle("Central Tendency of Mean Features in Malignant Diagnosis", fontsize = 15)
for i, col in enumerate(mean_cols):
    axes[i].boxplot(df[col][df['diagnosis'] == 'M'], patch_artist=True)
    axes[i].set_title(col)
plt.tight_layout()
plt.show()
```
![](https://github.com/ducnguyen8600/Breast-Cancer-Prediction-Project/blob/main/Mean%20Features%20Central%20tendency%20in%20Malignant%20Diagnosis.png)

### Conclusion: Descriptive statistic of Mean features

- **Mean**: Size-related features (such as radius, perimeter, and area) generally exhibit higher mean values compared to ratio-based features like smoothness or compactness. Those values accurately reflect the morphological characteristics of the dataset.

- **Standard deviation**: The standard deviations vary considerably across feature types

  - Size-related features show larger standard deviations, indicating substantial variability among observations.

  - Other features (smoothness, etc.) display smaller standard deviations, suggesting a more concentrated distribution.

- **Min-Max**: The min–max ranges indicate the presence of distant values, especially in size features. These may represent natural biological variation rather than measurement errors, implying the presence of natural outliers.

**When comparing the B and M classes:**
- **Malignant (M)** samples tend to have notably higher mean values for size-related features such as `radius`, `perimeter`, and `area`. A significant increase in `concave points` value was also observed in these samples

- **Benign (B)** samples show lower mean values, consistent with the typically smaller morphological characteristics of benign tissues.

- The `concavity` feature shows a large difference between the two classes. While the upper bound of this feature in the B class is around 0.12, it is approximately 0.34 in the M class. This difference may indicate a significant distinction between the two groups.

### Distribution of Mean features by Diagnosis
```python
for col in mean_cols:
  plt.figure(figsize=(10, 4))
  sns.kdeplot(data=df, x= col, hue='diagnosis', fill=True, common_norm = False)
  plt.title(f"Distribution of {col} by Diagnosis")
  plt.xlabel(f"{col}")
  plt.ylabel("Density")
  plt.show()
```
The plots show the distributions of all mean features. Here, I display only `radius_mean` and `fractal_dimension_mean` for comparision.
![](https://github.com/ducnguyen8600/Breast-Cancer-Prediction-Project/blob/main/Distribution%20of%20radius_mean%20by%20Diagnosis.png)
![](https://github.com/ducnguyen8600/Breast-Cancer-Prediction-Project/blob/main/Distribution%20of%20fractal_dimension%20by%20Diagnosis.png)

**Conclusion:**
**The distributions of features reveal significant differences between Benign and Malignant cases for all mean variables, with the exception of `fractal_dimension` and potentially `symmetry`.**

### Central tendency and Distribution of Worst Features
We performed the same steps as for the mean features and reached the same result. For more information, see the notebook here: [Breast_Cancer_Prediction](https://github.com/ducnguyen8600/Breast-Cancer-Prediction-Project/blob/main/Breast_Cancer_Prediction.ipynb)

### Convert the target column to a binary format (0 for benign, 1 for malignant)
```python
df['diagnosis'] = df['diagnosis'].map({'B':0, 'M': 1}).astype(int)
```

### Check Multicolinearity

#### Identify highly correlated features

**Pairplot for Mean and Worst features**
<p float="left">
  <img src="https://github.com/ducnguyen8600/Breast-Cancer-Prediction-Project/blob/main/Worst%20features%20pairplot.png" width="100%" />
  <img src="https://github.com/ducnguyen8600/Breast-Cancer-Prediction-Project/blob/main/Mean%20features%20pairplot.png" width="100%" />
</p>

**Heatmap for Mean and Worst features correlation**
![](https://github.com/ducnguyen8600/Breast-Cancer-Prediction-Project/blob/main/Heatmap%20for%20correlation.png)

**Conclusion**

We selected the highly correlated features (correlation > 0.8). Among these features, we identified two groups:

- Size-related features: `radius`, `perimeter`, `area`

- Concavity-related features: `concavity`, `concave points`, `compactness`

#### Select a feature that best represents the group
The correlations between compactness, concavity, concave points with the diagnosis were analyzed to identify and retain the feature exhibiting the strongest association
```python
## Mean feature:
for i in ['concave points_mean', 'concavity_mean', 'compactness_mean']:
  X = df[i]
  y = df['diagnosis']
  X = sm.add_constant(X)

  model = sm.Logit(y, X).fit()
  print(model.summary())
```
This is an example for Concavity-related mean features:

![](https://github.com/ducnguyen8600/Breast-Cancer-Prediction-Project/blob/main/Logistic%20regression%20(concave%20points).png)

Because these features all have p-values < 0.05, we will choose the one with the higher coefficient. So we go with `concave points_mean`.

We repeated the same procedure for the size-related mean features and worst features groups. For the remaining groups, we selected `radius_mean`, `radius_worst`, and `concave_points_worst`. For more information, see the notebook here: [Breast_Cancer_Prediction](https://github.com/ducnguyen8600/Breast-Cancer-Prediction-Project/blob/main/Breast_Cancer_Prediction.ipynb)

#### Exclude the remaining features in the group
**We removed the features perimeter, area, concavity, and compactness due to high multicollinearity, retaining only one representative variable for each correlated group.**

## 4. Analysis

### Build and Evaluate Logistic Regression Model
#### Mean features
**Build Model**
```python
X_mean_new = df[mean_cols].drop(columns = ['symmetry_mean', 'fractal_dimension_mean'])
y = df['diagnosis']

X_train_mean, X_test_mean, y_train_mean, y_test_mean = train_test_split(X_mean_new, y, test_size = 0.25, stratify = y, random_state = 42)

X_train_mean = sm.add_constant(X_train_mean)
X_test_mean = sm.add_constant(X_test_mean)

lr_model_mean_new = sm.Logit(y_train_mean, X_train_mean).fit()
lr_model_mean_new.summary()
```
![](https://github.com/ducnguyen8600/Breast-Cancer-Prediction-Project/blob/main/Mean%20Features%20Logistic%20Regression.png)

**Logistic Regression Mean Model Coefficients**
![](https://github.com/ducnguyen8600/Breast-Cancer-Prediction-Project/blob/main/Logistic%20Regression%20Mean%20Model%20Coefficients.png)

**Evaluation**

![](https://github.com/ducnguyen8600/Breast-Cancer-Prediction-Project/blob/main/Mean%20Features%20Logistic%20Regression%20Evaluation.png)

#### Worst features
**Build Model**
```python
X_worst_new = df[worst_cols].drop(columns = ['symmetry_worst', 'fractal_dimension_worst'])
y = df['diagnosis']

X_train_worst, X_test_worst, y_train_worst, y_test_worst = train_test_split(X_worst_new, y, test_size = 0.25, stratify = y, random_state = 42)

X_train_worst = sm.add_constant(X_train_worst)
X_test_worst = sm.add_constant(X_test_worst)

lr_model_worst_new = sm.Logit(y_train_worst, X_train_worst).fit()
lr_model_worst_new.summary()
```
![](https://github.com/ducnguyen8600/Breast-Cancer-Prediction-Project/blob/main/Worst%20Features%20Logistic%20Regression.png)

**Logistic Regression Worst Model Coefficients**
![](https://github.com/ducnguyen8600/Breast-Cancer-Prediction-Project/blob/main/Logistic%20Regression%20Worst%20%20Model%20Coefficients.png)

**Evaluation**

![](https://github.com/ducnguyen8600/Breast-Cancer-Prediction-Project/blob/main/Worst%20Features%20Logistic%20Regression%20Evaluation.png)

### Logistic Regression Model Conclusion
- We observed that `smoothness` and `concave points` have the strongest influence on the malignancy of cells, whereas `radius` and `texture` have a comparatively smaller effect.

- This observation is clinically reasonable, as not every large tumor is malignant, but a small tumor with low smoothness (high smoothness in this case) can be a strong indicator of malignancy.

- Including the constant feature in the model, which has a negative coefficient, is also reasonable as it helps to fine-tune the model.
  
### Evaluation Conclusion
- We can observe that the accuracy and F1-score on both the training and test sets are nearly identical, indicating that the model is not overfitting.

- The high evaluation scores are partly due to the fact that our dataset is relatively small.

- To enable more accurate clinical application, a sufficiently large dataset is required.

## 5. Conclusion
Based on the two models, it is evident that the likelihood of a cell being malignant is primarily influenced by four features: `radius`, `texture`, `concave points`, `smoothness`.

**Clinical Interpretation**

- Diagnosis is based on the most abnormal cell within the tissue sample.

- If even one malignant cell exists → the entire sample is considered cancerous.

**Model Selection**

*The worst-value model better reflects clinical practice → selected as the final predictive model*

**Model Interpretation**

- **Radius**: Each unit increase in `radius_worst` was associated with an estimated e^1.207-fold increase in the odds of cancer.

- **Texture**: Each unit increase in `texture_worst` was associated with an estimated e^0.286-fold increase in the odds of cancer.

- **Smoothness**: Each unit increase in `smoothness_worst` was associated with an estimated e^39.127-fold increase in the odds of cancer.

- **Concave points**: Each unit increase in `concave points_worst` was associated with an estimated e^28.508-fold increase in the odds of cancer.






