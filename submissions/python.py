import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("covid_case.csv")

print("\n----- First 5 Rows -----")
print(df.head())

print("\n----- Dataset Shape -----")
print(df.shape)

print("\n----- Column Names -----")
print(df.columns)

print("\n----- Missing Values -----")
print(df.isnull().sum())

df = df.fillna(0)

print("\n----- Total Records -----")
print(len(df))

print("\n----- Statistical Summary -----")
print(df.describe())

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

if 'Date' in df.columns and 'Confirmed' in df.columns:
    global_trend = df.groupby('Date')['Confirmed'].sum()

if 'Country' in df.columns and 'Confirmed' in df.columns:
    top10 = df.groupby('Country')['Confirmed'].sum().sort_values(ascending=False).head(10)

top5 = top10.head(5)

plt.figure(figsize=(10,5))
plt.plot(global_trend.index, global_trend.values)
plt.title("Global Trend of Confirmed COVID-19 Cases Over Time")
plt.xlabel("Date")
plt.ylabel("Total Confirmed Cases")
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.bar(top10.index, top10.values)
plt.title("Top 10 Countries by Confirmed COVID-19 Cases")
plt.xlabel("Country")
plt.ylabel("Total Confirmed Cases")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,8))
plt.pie(top5.values, labels=top5.index, autopct='%1.1f%%')
plt.title("Case Distribution of Top 5 Affected Countries")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(df['Confirmed'], df['Deaths'])
plt.title("Scatter Plot: Confirmed Cases vs Deaths")
plt.xlabel("Confirmed Cases")
plt.ylabel("Deaths")
plt.grid(True)
plt.show()

if 'Active' in df.columns:
    plt.figure(figsize=(8,6))
    plt.hist(df['Active'], bins=20)
    plt.title("Histogram of Active COVID-19 Cases")
    plt.xlabel("Active Cases")
    plt.ylabel("Frequency")
    plt.show()
else:
    print("\nNo 'Active' column found in dataset for Histogram.")