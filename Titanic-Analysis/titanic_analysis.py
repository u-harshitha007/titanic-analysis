import pandas as pd

df = pd.read_csv("titanic.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
df["Age"].fillna(df["Age"].mean(), inplace=True)
print(df.groupby("Sex")["Survived"].mean())

df.to_csv("cleaned_titanic.csv", index=False)

import matplotlib.pyplot as plt

# Survival by Gender
survival_by_gender = df.groupby("Sex")["Survived"].mean()

survival_by_gender.plot(kind="bar")

plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")

plt.show()

# Survival by Passenger Class Visualization

import matplotlib.pyplot as plt

survival_by_class = df.groupby("Pclass")["Survived"].mean()

survival_by_class.plot(kind="bar")

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.show()
plt.figure()

plt.hist(df["Age"])

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.show()

plt.figure()

fare_by_class = df.groupby("Pclass")["Fare"].mean()

fare_by_class.plot(kind="bar")

plt.title("Average Fare by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Average Fare")

plt.show()

import seaborn as sns

plt.figure()

sns.heatmap(
    df[["Survived","Age","Fare","Pclass"]].corr(),
    annot=True
)

plt.title("Correlation Heatmap")

plt.show()