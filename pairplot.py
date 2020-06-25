import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="ticks", color_codes=True)
dataset = pd.read_csv('training_set.csv')
dataset.describe(include='all')
sns_plot = sns.pairplot(dataset, hue='CLASS',height=2.5)
plt.show()
