import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df = pd.read_csv("../im/Average_daily_airQualityInfo_2018.csv", engine='python')
df.rename(columns={"측정일자": "cdate", "권역코드": "acode", "권역명": "aname", "측정소코드": "scode",
                   "측정소명": "sname", "미세먼지(㎍/㎥)": "fdust", "초미세먼지(㎍/㎥)": "ufdust",
                   "오존(ppm)": "ozone", "이산화질소농도(ppm)": "nd", "일산화탄소농도(ppm)": "cm", "아황산가스농도(ppm)": "sagas"},
          inplace=True)
# print(df.columns)

cleaning_filter = ((df['fdust'] == 0) & (df['ufdust'] == 0) & (df['ozone'] == 0) & (
        df['nd'] == 0) & (df['cm'] == 0) & (df['sagas'] == 0))
# cleaning_df = cleaning_df.drop(cleaning_df.index)
cleaning_df = df[cleaning_filter]
cleaning_df = df.drop(cleaning_df.index)
# df_delete = df[df.fdust != 0]
# print(df_delete)
cleaning_df.to_csv("data1.csv", encoding='ms949', index=False)

"""
def read_data(file):
    f = open(file, 'r')
    data = csv.reader(f)
    for line in data:
        print(line)
    return data
"""
