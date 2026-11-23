# Step-1: Import libraries
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestCLassifier
from sklearn.svm import SVC
from sklearn.metrics import (f1_score,accuracy_score,classification_report,confusion_matrix)
# Step-2: Load Dataset
data = load_breast_cancer()
X = pd.DataFrame(data,data,columns=data.features_names)
y = pd.Series(data.target,name="target")
print("Dataset shape:".X.shape)
print("\nFirst 5 rows:")
print(X.head())
print("\nTarget Values:")
print(y.value_counts())
# Step-3: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
# Step-4: Define classification algorthms
models - {"Logistic Regression":LogisticRegression(max_iter=5000),
          "KNN":KNeighborsClassifier(),
          "Decision Tree":DecisionTreeClassifier(random_state=42),
          "Random Forest":DecisionTreeClassifier(random_state=42),
          "SVM":SVC()
         }
# Step-5: Define hyperparameter grids
param_grids = {"Logistic Regression":{"model_C":[0.1,1,10],"model_solver":["liblinear"]},
               "KNN":{"model_n_neighbors":[3,5,7,9],"model_weights":["uniform","distance"]},
               "Decision Tree":{"model_max_depth":[None, 3, 5, 10],"model_min_samples_split":[2,5,10]
               "Random Forest":{"model_n_estimators":[100,200],"model_max_depth":[None,5,10],"model_min_samples_split":[2,5]},
               "SVM":{model_C":[0.1,1,10],"model_kernel":["linear", "rbf"],"model_gamma":["scale","auto"]}}
# Step-6: Run 
               
