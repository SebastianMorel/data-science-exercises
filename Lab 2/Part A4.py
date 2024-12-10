import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
"""
1. Correlation between variables.
"""
corr = df.corr()
sns.heatmap(corr, annot=corr.round(2), annot_kws={"size":9})
plt.title('Correlation between variables')
plt.show()
plt.clf()
"""
2. Most expensive suburbs to live in by median price.
"""
groupedSuburbs = df.groupby('Suburb')[['Price']].median()
print(groupedSuburbs.sort_values(by=['Price'],ascending=False))
"""
3. Methods that have the highest average price.
"""
methods = df.groupby('Method')[['Price']].mean()
plt.ticklabel_format(style = 'plain')
plt.bar(methods.index, methods['Price'])
plt.title('Average price for each method')
plt.xlabel('Method')
plt.ylabel('Price')
plt.show()