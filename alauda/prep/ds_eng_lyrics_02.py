import pandas as pd
import numpy as np

csv_file_path = 'C:/Dev/ds/tcc_ceds_music.csv'
prep_out = 'C:/Dev/ds/eng_lyrics_02.txt'
prep_train_out = 'C:/Dev/ds/eng_lyrics_train_02.txt'
prep_valid_out = 'C:/Dev/ds/eng_lyrics_valid_02.txt'
prep_test_out = 'C:/Dev/ds/eng_lyrics_test_02.txt'

pd_data = pd.read_csv(csv_file_path)

pd_data = pd_data[['release_date', 'lyrics']]
pd_data = pd_data[pd_data['release_date'].notnull()]
pd_data = pd_data[pd_data['lyrics'].notnull()]
pd_data = pd_data.rename(columns={'release_date': 'age'})
pd_data['age'] = pd_data['age'].astype(str).str[0]
pd_data['lyrics'] = pd_data['lyrics'].str.casefold()

print(pd_data)
rows_num = pd_data.shape[0]
print(rows_num)

pd_data = pd_data.sample(rows_num)
print(pd_data)
pd_data.to_csv(prep_out)

train_df, valid_df, test_df = np.split(pd_data, [int(.75 * rows_num), int(.9 * rows_num)])

train_df.to_csv(prep_train_out)
valid_df.to_csv(prep_valid_out)
test_df.to_csv(prep_test_out)
