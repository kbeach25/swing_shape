from pybaseball import statcast, playerid_reverse_lookup

def getData():
    raw_df = statcast('2025-03-27', str(date.today()))
    
    ids = raw_df['batter'].dropna().unique()
    id_df = playerid_reverse_lookup(ids, key_type='mlbam')

    id_to_name = {
    row['key_mlbam']: f"{row['name_first']} {row['name_last']}"
    for _, row in id_df.iterrows()
    }

    raw_df['batter_name'] = raw_df['batter'].map(id_to_name)
    
    return raw_df

# Statcast data from statcast
df = getData()

df.to_csv("season_data.csv")
