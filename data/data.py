import pandas as pd

df = pd.read_csv(r"C:\Symbiotic-twin\data\iot_telemetry_data.csv")
print(df.head())
print(df.isnull().sum())