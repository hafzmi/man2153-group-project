import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
# from warnings import simplefilter
# from sklearn.exceptions import ConvergenceWarning

# simplefilter("ignore", category=ConvergenceWarning)

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