import pandas as pd
import seaborn as sns
from IPython.display import display_html
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sns.set(style="darkgrid")


def stats_table(column1, column2):
    normilize_crosstab = pd.crosstab(column1,
                                     column2,
                                     normalize='index',
                                     margins=True)

    crosstab = pd.crosstab(column1,
                           column2,
                           margins=True)

    return normilize_crosstab, crosstab


def visualize_categorical(column_x, column_y,
                          height=5, rotate=30, col_wrap=None):
    normilize_crosstab, crosstab = stats_table(column1=column_y,
                                               column2=column_x)

    normilize_crosstab = round(normilize_crosstab * 100, 2)
    normilize_stat = normilize_crosstab.stack().reset_index()

    sns.set_palette("deep")
    g = sns.catplot(x=column_x.name,
                    y=0,
                    col=column_y.name,
                    col_wrap=col_wrap,
                    data=normilize_stat,
                    kind="bar",
                    height=height)
    g.set_xticklabels(rotation=rotate)

    display_df = display_html((normilize_crosstab.to_html() +
                               crosstab.to_html()).replace('table',
                                                           'table style="display:inline"'),
                              raw=True)
    return display_df, g


def visualize_continuous(column_x, column_y, df,
                         height=5, rotate=30,
                         col=None, col_wrap=None):
    sns.set_palette("deep")
    g = sns.catplot(x=column_x,
                    y=column_y,
                    col=col,
                    col_wrap=col_wrap,
                    data=df,
                    kind="box",
                    height=height)
    g.set_xticklabels(rotation=rotate)

    return


def visualize_tsn_pca(df_decompose, y_set, type='tsne'):
    f, ax = plt.subplots(figsize=(5, 5))

    blue_patch = mpatches.Patch(color='#0A0AFF',
                                label='No Fraud')
    red_patch = mpatches.Patch(color='#AF0000',
                               label='Fraud')

    ax.scatter(df_decompose[:, 0],
               df_decompose[:, 1],
               c=(y_set == 0),
               cmap='coolwarm',
               label='No Fraud',
               linewidths=2)
    ax.scatter(df_decompose[:, 0],
               df_decompose[:, 1],
               c=(y_set == 1),
               cmap='coolwarm',
               label='Fraud',
               linewidths=2)
    ax.set_title(type,
                 fontsize=14)

    ax.grid(True)

    ax.legend(handles=[blue_patch, red_patch])


def visualize_roc_score(df, names, type_sampl):
    fig = plt.figure(figsize=(12, 10))
    plt.title('Comparison of Classification Algorithms for {}'.format(type_sampl))
    plt.xlabel('Algorithm')
    plt.ylabel('Roc Auc')
    plt.boxplot(df)
    ax = fig.add_subplot(111)
    ax.set_xticklabels(names)
    plt.show()
