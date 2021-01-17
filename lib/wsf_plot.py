import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

def wsf_plot_corr_matrix(df_corr):
  _ = plt.subplots(figsize=(20, 15))
  
  sns.set(style="white")

  cmap = sns.diverging_palette(220, 10, as_cmap=True)
  mask = np.triu(np.ones_like(df_corr, dtype=np.bool)) # upper triangle of the matrix, all values are 1
  sns.heatmap(df_corr, mask=mask, cmap=cmap, vmax=.3, center=0,
              square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
  plt.show()

def wsf_plot_violin(df, x, y):
  fig = px.violin(df, x=x, y=y, color=x, box=True, points="all", hover_data=df.columns)
  fig.show()

def wsf_plot_scores(df_train_results):
  plt.plot(df_train_results['mean_test_score'])
  plt.show()  

def wsf_plot_scores_by_param(df_train_results, param):
  df_sorted = df_train_results.sort_values(param, ignore_index=True)

  plt.plot(df_sorted['mean_test_score'])
  plt.xticks(np.arange(df_sorted.shape[0]),df_sorted[param])
  plt.show()

def wsf_plot_train_results(results, column_name):
  plt.xticks(np.arange(100))
  for (name, df) in results.items():
    plt.plot(df[column_name], label=name)

  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.ylabel(column_name)
  plt.show()

def wsf_plot_bar_chart(dict, title):
  values = dict.values()
  for idx, val in enumerate(values):
    plt.bar(idx, val)

  plt.xticks(np.arange(len(values)), labels=list(dict.keys()))
  plt.ylabel(title)
  plt.show()