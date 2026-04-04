import pandas as pd

df = pd.read_csv('data/dataset1.csv')

# countries = df['Country Code'].unique()
countries = ['JOR', 'EGY', 'DZA']


target_df = (df[(df['Series Code'] == 'SL.EMP.SMGT.FE.ZS')])
for c in countries:
    sdf = target_df[target_df['Country Code'] == c]
    sdf = sdf[sdf.isna().sum(axis=1) <= 25]
    if len(sdf) > 0:
        print(sdf.iloc[0])