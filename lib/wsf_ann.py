import time
import uuid
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

DIR_SAVED_WEIGHTS = 'tmp-ann-weights'

"""Builds an ANN.

    Parameters
    ----------
    layers: list
        Each item is a tuple with 2 elements: number of nodes, and activation function.
    input_dim: int
        Input data dimension.

    Return values
    ----------
    The neural network.
"""    
def wsf_create_ann(layers, input_dim):
  model = Sequential()

  # input layer
  (input_layer_nodes, input_layer_activation) = layers[0]
  model.add(Dense(input_layer_nodes, activation=input_layer_activation, input_dim=input_dim))

  # multiple hidden layers
  for i in np.arange(1, len(layers)):
    (n_nodes, activation) = layers[i]
    model.add(Dense(n_nodes, activation=activation))
  
  # output layer
  model.add(Dense(1, activation='linear'))

  return model


"""Loops through a list of ANNs, trains them and evaluates the results to find the best model.

    Parameters
    ----------
    anns: list
        Each item is a candidate model.
    X: DataFrame
        Input data for training.
    y: Series
        Labels for X.
    cv: int
        Number of folds in k-fold cross validation.
    epochs, batch_size, validation_split:
        Parameters to pass through to model.fit()
    
    Return values
    ----------
    A data frame containing training results, and the index of the best model in the list.
"""    
def wsf_find_best_ann_cv(anns, X, y, cv=3, epochs=20, batch_size=1000, validation_split=0.2):
  results = []
  best_mean_score = 0
  best_model_index = 0

  for i in np.arange(len(anns)):
    model_info = wsf_ann_cv(anns[i], X, y, cv, epochs, batch_size, validation_split)
    model_info['model_id'] = 'model-' + str(i)
    if (model_info['mean_test_score'] > best_mean_score):
      best_mean_score = model_info['mean_test_score']
      best_model_index = i
    results.append(model_info)
  
  return pd.DataFrame(results), best_model_index


"""Trains an ANN using k-fold cross validation.

    Parameters
    ----------
    ann: model
        The neural network to train.
    X: DataFrame
        Input data for training.
    y: Series
        Labels for X.
    cv: int
        Number of folds in k-fold cross validation.
    epochs, batch_size, validation_split:
        Parameters to pass through to model.fit()
    
    Return values
    ----------
    A dictionary containing training results: splitX_test_score, mean_test_score, mean_fit_time.
"""    
def wsf_ann_cv(ann, X, y, cv, epochs, batch_size, validation_split):
  total_scores = 0
  total_fit_time = 0
  split_index = 0
  results = {}

  kf = KFold(n_splits = cv, shuffle=True)
  for train_index, val_index in kf.split(X):
    X_fit, y_fit = X.iloc[train_index], y.iloc[train_index]
    X_val, y_val = X.iloc[val_index], y.iloc[val_index]

    model = clone_model(ann)
    fit_time = wsf_ann_fit_save_best(model, X_fit, y_fit, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    y_pred = model.predict(X_val)
    score = r2_score(y_val, y_pred)

    total_scores += score
    total_fit_time += fit_time
    results['split' + str(split_index) + '_test_score'] = score
    split_index += 1

  results['mean_test_score'] = total_scores / cv
  results['mean_fit_time'] = total_fit_time / cv
  return results

"""Trains an ANN and retains the best weights found during the training process.

    Parameters
    ----------
    ann: model
        The neural network to train.
    X: DataFrame
        Input data for training.
    y: Series
        Labels for X.
    epochs, batch_size, validation_split:
        Parameters to pass through to model.fit()
    
    Return values
    ----------
    A data frame containing training results, and the index of the best model in the list.
"""    
def wsf_ann_fit_save_best(model, X, y, epochs=20, batch_size=1000, validation_split=0.2):
  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

  cp_file_path = DIR_SAVED_WEIGHTS + '/' + str(uuid.uuid4()) + '.hdf5' 
  cp = ModelCheckpoint(cp_file_path, monitor='val_mean_squared_error', verbose = 1, save_best_only = True, mode ='auto')

  time_start = time.time()
  model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[cp])
  fit_time = time.time() - time_start

  model.load_weights(cp_file_path)

  return fit_time