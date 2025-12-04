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
  
