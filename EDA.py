import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transform_data import df_EDA

plt.rcParams.update({'font.size': 8})


def horizontal_bar_price(column, figsize=(10, 5)):
    plt.figure(figsize=figsize, dpi=80)
    pl = sns.barplot(x=df_EDA["Price"], y=column, estimator=np.median, orient='h')
    pl.set_title(column.name)
    pl.set_xlabel("Price of flight")
    pl.set_ylabel(column.name)
    plt.savefig(f'graphs/{column.name}.png')


horizontal_bar_price(df_EDA["Airline"], (18, 5))

horizontal_bar_price(df_EDA["season"])

horizontal_bar_price(df_EDA["Dep_Time_Period"])

horizontal_bar_price(df_EDA["Arrival_Time_Period"])

fig, axes = plt.subplots(2, 2, figsize=(15, 7))
fig.tight_layout(pad=3.0)

sns.barplot(data=df_EDA, x=df_EDA["Additional Info_Business class"],
            y=df_EDA["Price"], ax=axes[0, 0], estimator=np.median)
axes[0, 0].set_title("Business class")
axes[0, 0].set_xlabel("Price of flight")
axes[0, 0].set_ylabel(None)

sns.barplot(data=df_EDA, x=df_EDA["Additional Info_Change airports"],
            y=df_EDA["Price"], ax=axes[0, 1], estimator=np.median)
axes[0, 1].set_title("Change of airports")
axes[0, 1].set_xlabel("Price of flight")
axes[0, 1].set_ylabel(None)

sns.barplot(data=df_EDA, x=df_EDA["Additional Info_In-flight meal not included"],
            y=df_EDA["Price"], ax=axes[1, 0], estimator=np.median)
axes[1, 0].set_title("In-flight meal not included")
axes[1, 0].set_xlabel("Price of flight")
axes[1, 0].set_ylabel(None)

sns.barplot(data=df_EDA, x=df_EDA["Additional Info_No check-in baggage included"],
            y=df_EDA["Price"], ax=axes[1, 1], estimator=np.median)
axes[1, 1].set_title("No check-in baggage included")
axes[1, 1].set_xlabel("Price of flight")
axes[1, 1].set_ylabel(None)

plt.savefig(f'graphs/binary_features.png')
