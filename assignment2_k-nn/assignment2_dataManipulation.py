import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 하나의 행에서 데이터를 모두 비교하고 제일 낮은 등급으로 나온 것이 airquality가 된다

airquality = [['best', 15, 8, 0.02, 0.02, 1, 0.01], ['better', 30, 15, 0.03, 0.03, 2, 0.02],
              ['good', 40, 20,0.06, 0.05, 5.5, 0.04], ['normal', 50, 25, 0.09, 0.06, 9, 0.05],
              ['bad', 75, 37, 0.12, 0.13, 12, 0.1], ['worse', 100, 50, 0.15, 0.2, 15, 0.15],
              ['serious', 150, 75, 0.38, 1.1, 32, 0.6]]

df = pd.read_csv('../assignment1_regression/data1.csv', sep=',', engine='python')
df['airquality'] = 'worst'

for i in reversed(airquality):
    df_filter = df[(df[df.columns[5]] < i[1]) & (df[df.columns[6]] < i[2]) & (df[df.columns[7]] < i[3]) & (df[df.columns[8]] < i[4]) & (df[df.columns[9]] < i[5]) & (df[df.columns[10]] < i[6])]
    df.loc[df_filter.index, 'airquality'] = i[0]

del df['ufdust']

df.to_csv('data2.csv', encoding='ms949', index=False)