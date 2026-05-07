import pandas as pd

df1 = pd.read_csv("preprocessed_data/github-data/labeled_data.csv")
df2 = pd.read_csv("preprocessed_data/jigsaw-data/train.csv")
df3 = pd.read_csv("preprocessed_data/jigsaw-data/test.csv")

print("Data Frame 1:")
print(df1.head())
print(df1.shape)

print("Data Frame 2:")
print(df2.head())
print(df2.shape)

print("Data Frame 3:")
print(df3.head())
print(df3.shape)