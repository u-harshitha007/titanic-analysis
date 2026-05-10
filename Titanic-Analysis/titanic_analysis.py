import pandas as pd

df = pd.read_csv("titanic.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
df["Age"].fillna(df["Age"].mean(), inplace=True)
print(df.groupby("Sex")["Survived"].mean())

df.to_csv("cleaned_titanic.csv", index=False)