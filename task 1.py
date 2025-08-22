
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("data.csv")
print(df.head())
print("Summary Stats:\n", df.describe())
print("Null Values:\n", df.isnull().sum())
print("Average of column X:", df['ColumnName'].mean())
df['ColumnName'].value_counts().plot(kind='bar')
plt.title("Bar Chart of ColumnName")
plt.show()
plt.scatter(df['Column1'], df['Column2'])
plt.xlabel("Column1")
plt.ylabel("Column2")
plt.title("Scatter Plot")
plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
