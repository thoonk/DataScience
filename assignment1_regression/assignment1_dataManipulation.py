import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df = pd.read_csv('data1.csv', sep=',', engine='python')

for i in df.columns[:5]:
    del df[i]

scaler = MinMaxScaler()
df[:] = scaler.fit_transform(df[:])
print(df)
df.to_csv('data2.csv', encoding='ms949', index=False)
