import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

def wsf_get_neg_sales(df_data):
  df_neg = df_data[df_data['Weekly_Sales'] < 0]
  df_no_md = pd.DataFrame({
      'No_MarkDown': df_neg['MarkDown1'] + df_neg['MarkDown2'] + df_neg['MarkDown3'] + df_neg['MarkDown4'] + df_neg['MarkDown5'] == 0
  }) 
  return pd.concat([df_neg, df_no_md], axis=1)

def wsf_find_best_model_cv(regressor, params, X_train, y_train, no_of_models=5, cv=3):
  search = RandomizedSearchCV(
    estimator = regressor, 
    param_distributions = params, 
    n_iter = no_of_models, 
    cv = cv, 
    random_state=0, 
    n_jobs = -1
  )
  search.fit(X_train, y_train)

  train_results = pd.DataFrame(search.cv_results_)
  best_model = search.best_estimator_

  return train_results, best_model 

def wsf_predict_and_eval(model, X, y):
  y_pred = model.predict(X)
  y_pred.resize(len(y_pred),) # necessary for ANNs
  return wsf_weighted_mean_abs_err(X, y, y_pred)

def wsf_weighted_mean_abs_err(X, y, y_pred): 
  # weight: 5 for holiday, 1 for non-holiday
  w = X['IsHoliday'].apply(lambda val: 5 if val == 1 else 1)
  
  wmae_score = np.sum(w * np.abs(y - y_pred), axis=0) / np.sum(w)
  return wmae_score
