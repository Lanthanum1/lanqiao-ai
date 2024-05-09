import pandas as pd

songs = pd.read_csv('songs_origin.csv')

# 对于数据集中的缺失值，以其所在列的均值进行填充。
songs = songs.fillna(songs.mean())

# 删除大于 1 或小于 0 的行
songs = songs[songs['acousticness_yr'].between(0, 1)]

# 删除重复行
songs.drop_duplicates(inplace=True)

# 保存处理后的数据集
songs.to_csv('songs_processed.csv', index=False)