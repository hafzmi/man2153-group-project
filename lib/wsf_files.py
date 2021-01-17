import joblib
import pandas as pd
from keras.models import model_from_json

def wsf_save_train_dataset(X, y):
  X.to_csv('X_train.csv', index=False)
  y.to_csv('y_train.csv', index=False)

def wsf_load_train_dataset():
  X = pd.read_csv('X_train.csv')
  y_df = pd.read_csv('y_train.csv')
  return X, y_df['Weekly_Sales']

def wsf_save_test_dataset(X, y):
  X.to_csv('X_test.csv', index=False)
  y.to_csv('y_test.csv', index=False)

def wsf_load_test_dataset():
  X = pd.read_csv('X_test.csv')
  y_df = pd.read_csv('y_test.csv')
  return X, y_df['Weekly_Sales']

def wsf_save_training_results(df_results, name):
  df_results.to_csv(name + '-training.csv', index=False)

def wsf_load_training_results(name):
  return pd.read_csv(name + '-training.csv')

def wsf_save_best_model(model, name):
  joblib.dump(model, name + '-best.model')

def wsf_load_best_model(name):
  return joblib.load(name + '-best.model')

def wsf_save_best_ann(ann):
  # save weights to .h5 file
  ann.save_weights("ann-best.h5")

  # save model to json file
  with open("ann-best.json", "w") as json_file:
      json_file.write(ann.to_json())

def wsf_load_best_ann():
  # load model from json file
  json_file = open('ann-best.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()

  loaded_model = model_from_json(loaded_model_json)

  # load weights into new model
  loaded_model.load_weights("ann-best.h5")

  return loaded_model