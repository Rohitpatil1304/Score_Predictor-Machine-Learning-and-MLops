import pandas as pd
import pickle
import numpy as np


df = pickle.load(open('data/processed/dataset_level2.pkl' , 'rb'))

cities = np.where(df['city'].isnull(), df['venue'].str.split().apply(lambda x:x[0]), df['city'])

df['city'] = cities

df.drop(columns=['venue'],inplace = True)

eligible_cities = df['city'].value_counts()[df['city'].value_counts() > 600].index.tolist()


df = df[df['city'].isin(eligible_cities)]

df['current_score'] = df.groupby('match_id')['runs'].cumsum()

df['over'] = df['ball'].apply(lambda x:str(x).split(".")[0])
df['ball_no'] = df['ball'].apply(lambda x:str(x).split(".")[1])

df['balls_bowled'] = (df['over'].astype('int')*6) + df['ball_no'].astype('int')


df['balls_left'] = 120 - df['balls_bowled']
df['balls_left'] = df['balls_left'].apply(lambda x:0 if x<0 else x)

df['player_dismissed'] = df['player_dismissed'].apply(lambda x:0 if x=='0' else 1)
df['player_dismissed'] = df['player_dismissed'].astype('int')
df['player_dismissed'] = df.groupby('match_id')['player_dismissed'].cumsum()
df['wickets_left'] = 10 - df['player_dismissed']

df['crr'] = (df['current_score']*6)/df['balls_bowled']


groups = df.groupby('match_id')

match_ids = df['match_id'].unique()
last_five = []

for id in match_ids:
    last_five.extend(
        groups.get_group(id)['runs']
        .rolling(window=30)
        .sum()
        .values
        .tolist()
    )

df['last_five'] = last_five


final_df = df.groupby('match_id').sum()['runs'].reset_index().merge(df, on='match_id')

final_df = final_df[['batting_team' , 'bowling_team' , 'city' ,'current_score' , 'balls_left' , 'wickets_left' , 'crr' , 'last_five' , 'runs_x']]

final_df.dropna(inplace = True)

final_df = final_df.sample(final_df.shape[0])

pickle.dump(final_df,open('data/interim/dataset_level3_feature_ready.pkl','wb'))