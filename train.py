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
# Step-6: Run GridSearchCV for every alogorithm
results = []
best_models = {}
for model_name, model in models.items():
   print("\n" + "=" * 60)
   print("Running:", model_name)
   print("=" * 60)
   pipeline = Pipeline([{"scaler",StandardScaler()),("model",model)])
   grid_search = GridSearchCV(estimator=pipeline,param_grid=param_grids[model_name],scoring="f1",cv=5,n_jobs=-1)
   grid_search.fit(X_train, y_train)
   best_models[model_name] = grid_search.best_estimator_results.append({"Model":model_name,"Best F! score":grid_search.best_score_,"Best parameters":grid_search.best_params_})
   print("Best F! Score:",round(grid_search.best_score_,4))
   print("Best Parameters:",grid_search.best_params_)
   # Step-7: compare all models
   results_df = pd.DataFrame(results)
   results_df = results_df.sort_values(by="Best F1 Score",ascending=False)
   print("\n" + "=" * 60)
   print("Model comparison")
   print("=" * 60)
   print(results_df[["Model","Best F! Score"]].to_string(index=False))
   # Step-8: Select the best model
   best_model_name = results_df.iloc[0]["Model"]
   best_model = best_models[best_model_name]
   best_cv_f1 = results_df.iloc[0]["Best F1-score:")
   print("\nBest model based on CV F1-score:")
   print(best_model_name)
   print("\nBest CV F1-Score:")
   print(round(best_cv_f1,4))
   # Step-9: Evaluate on unseen test data
   y_pred = best_model.predict(X_test)
   y_pred = best_model.predict(X_test)
   print("\n" + "=" * 60)
   print("Final Model Evaluation")
   print("=" * 60)
   print("Accuracy"", f1_score(y_test, y_pred))
   print("\nClassification Report:")
   print(classification_report(y_test, y_pred, target_names=data.target_names))
   print(("\nConfusion Matrix:")
   print(confusion_matrix(y_test, y_pred))
   # Step-10: Save the best data model 
   joblib.dump(best_model, "best_model.joblib")
   print("\nBest model saved successdfully!")
   print("Model:", best_model_name)
               
