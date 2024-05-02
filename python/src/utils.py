from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import pearsonr

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# https://stackoverflow.com/questions/63416894/correlation-values-in-pairplot
def reg_coef(
    x,
    y,
    label=None,
    color=None,
    **kwargs
    ):
    ax = plt.gca()
    r,p = pearsonr(x,y)
    text_color = plt.cm.RdYlBu((r + 1) / 2)  # Map correlation value to color
    text_size = abs(r) * 20  # Scale text size based on absolute correlation value
    ax.annotate(
        'r = {:.2f}'.format(r),
        xy=(0.5, 0.5),
        xycoords='axes fraction',
        ha='center',
        # color=text_color,
        fontsize=text_size
    )
    ax.set_axis_off()
    
def metrics_score(
    model:BaseEstimator,
    actual:pd.Series,
    predicted:np.array
    ):
    try:
        print(classification_report(actual, predicted, labels=model.classes_))
    except:
        print(classification_report(actual, [model.classes_[i] for i in predicted], labels=model.classes_))

    cm = confusion_matrix(actual, predicted, labels=model.classes_)
    
    plt.figure(figsize=(8,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='g',
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        cmap='Blues',
    )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
