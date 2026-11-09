import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
def create_pipeline():
  # define which features are numeric vs categorical
  numeric_features = ['Age','Income']
  categorical_features = ['City','Occupation']
  # preprocessing for numerical data(impute missing, then scale)
  numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
  # preprocessing for categorical data
  categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
  # combine into a single ColumnTransformer
  preprocessor = ColumnTransformer(
    transformers=[
      ('num', numeric_transformer, numeric_features),
      ('cat', categorical_transformer, categorical_features)])
  return preprocessor
  if __name__ == '__main__':
    print("Preprocessing pipeline template ready.")
