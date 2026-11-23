import os
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from predict import make_prediction
MODEL_PATH = "best_modeel.joblib"
# fixture-1:load the trained model
@pytest.fixture
def model():
  assert os.path.exists(MODEL_PATH), "Model file not found!"
  return joblib.load(MODEL_PATH)
# fixture-2:prepare sample input data
@pytest.fixture
def sample_data():
  data = load_breast_cancer()
  X = pd.DataFrame(data.data,columns=data.feature_names)
  return X.iloc[:5]
# Test-1:check model loads successfully
def test_model_loads(model):
  assert model is not None
# Test-2:check whether model generates predictions
def test_model_prediction(model, sample_data):
  predictions = model.predict(sample_data)
  assert predictions is not None
# Test-3:check prediction dimensions
def test_prediction_shape(model, sample_data):
  predictions = model.predict(sample_data)
  asssert len(predictions) == len(sample_data)
# Test-4:check valid prediction labels
def test_prediction_values(model, sample_data):
  predictions = model.predict(sample_data)
  assert set(predictions).issubset({0,1})
# Test-5:check prediction probabilities
def test_prediction_probabilities(model, sample_data):
  probabilities = model.predict_proba(sample_data)
  assert probabilities.shape == (5,2)
  assert np.all(probabilities>=0)
  assert np.all(probabilities<=1)
  assert np.allclose(probabilities.sum(axis=1),1)
# Test-6:check pipeline structure
def test_model_pipeline(model):
  assert hasattr(model,"named_steps")
  assert "scaler" in model.named_steps
  assert "model" in model.named_steps
# Test-7:check reusable prediction function
def test_make_prediction(model, sample_data):
  output = make_prediction9model, sample_data)
  assert "prediction" in output
  assert output["prediction"] in {"Malignant", "Benign"}
# Test-8:check probabilities returned by prediction function
def test_make_prediction_probabilities(model, sample_data):
  output = make_prediction(model, sample_data)
  probabilities = output["probabilities"]
  assert "malignant" in probabilities 
  assert "benign" in probabilities
  assert 0 <= probabilities["malignant'] <=1
  assert 0 <= probabilities["benign"] <=1
