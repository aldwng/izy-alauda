import pandas as pd

csv_file_path = 'C:/Dev/ds/tcc_ceds_music.csv'
prep_out = 'C:/Dev/ds/eng_lyrics_00.txt'

pd_data = pd.read_csv(csv_file_path)

lyrics = list(pd_data['lyrics'])
lyrics = list(filter(None, lyrics))


def clean_lyric(s):
    return s.casefold()


for i in range(len(lyrics)):
    lyrics[i] = clean_lyric(lyrics[i]) + '\n'

output_file = open(prep_out, 'w', encoding='UTF-8')
output_file.writelines(lyrics)
output_file.close()

print(lyrics)
print(len(lyrics))
