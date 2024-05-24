import pandas as pd

df = pd.read_csv('song_origin.csv')

df = df.fillna(df.mean())

df = df[(df['acousticness_yr'] >= 0) & (df['acousticness_yr'] <= 1)]

df = df.drop_duplicates()

df.to_csv('song_processed.csv', index=False)