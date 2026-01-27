import pickle 
import pandas as pd
import numpy as np


df = pickle.load(open('data/raw/dataset_level_new.pkl','rb'))

backup = df

df.drop(columns=[
    'meta.data_version',
    'meta.created',
    'meta.revision',
    'info.outcome.bowl_out',
    'info.bowl_out',
    'info.supersubs.South Africa',
    'info.supersubs.New Zealand',
    'info.outcome.eliminator',
    'info.outcome.result',
    'info.outcome.method',
    'info.neutral_venue',
    'info.match_type_number',
    'info.outcome.by.runs',
    'info.outcome.by.wickets'
],inplace=True)


required_columns = [
    "innings",
    "info.dates",
    "info.gender",
    "info.match_type",
    "info.outcome.winner",
    "info.overs",
    "info.player_of_match",
    "info.teams",
    "info.toss.decision",
    "info.toss.winner",
    "info.umpires",
    "info.venue",
    "match_id",
    "info.city"
]

df = df[required_columns]

print(df['info.gender'].value_counts())

df = df[df['info.gender'] == 'male']
df.drop(columns=['info.gender'] , inplace = True)

df.drop(columns=['info.match_type'] , inplace= True)

df = df[df['info.overs'] == 20 ]
df.drop(columns=['info.overs'] , inplace = True)


pickle.dump(df,open('data/raw/dataset_level1.pkl','wb'))

matches = pickle.load(open('data/raw/dataset_level1.pkl','rb'))

count = 1
dfs = []   # collect DataFrames here

for index, row in matches.iterrows():

    if count in [75,108,150,180,268,360,443,458,584,748,982,1052,1111,1226,1345]:
        count += 1
        continue

    ball_of_match = []
    batsman = []
    bowler = []
    runs = []
    player_of_dismissed = []
    teams = []
    batting_team = []
    match_id = []
    city = []
    venue = []

    for ball in row['innings'][0]['1st innings']['deliveries']:
        for key in ball.keys():
            match_id.append(count)
            batting_team.append(row['innings'][0]['1st innings']['team'])
            teams.append(row['info.teams'])
            ball_of_match.append(key)
            batsman.append(ball[key]['batsman'])
            bowler.append(ball[key]['bowler'])
            runs.append(ball[key]['runs']['total'])
            city.append(row['info.city'])
            venue.append(row['info.venue'])
    
            wicket = ball[key].get('wicket')
            if isinstance(wicket, dict):
                player_of_dismissed.append(wicket.get('player_out', '0'))
            elif isinstance(wicket, list) and len(wicket) > 0:
                player_of_dismissed.append(wicket[0].get('player_out', '0'))
            else:
                player_of_dismissed.append('0')
    loop_df = pd.DataFrame({
        'match_id': match_id,
        'teams': teams,
        'batting_team': batting_team,
        'ball': ball_of_match,
        'batsman': batsman,
        'bowler': bowler,
        'runs': runs,
        'player_dismissed': player_of_dismissed,
        'city': city,
        'venue': venue
    })

    dfs.append(loop_df)
    count += 1

# Final DataFrame
delivery_df = pd.concat(dfs, ignore_index=True)

def bowl(row):
    for team in row['teams']:
        if team != row['batting_team']:
            return team
        
delivery_df['bowling_team'] = delivery_df.apply(bowl,axis=1)

delivery_df.drop(columns=['teams'],inplace=True)

teams = [
    'Australia',
    'India',
    'Bangladesh',
    'New Zealand',
    'South Africa',
    'England',
    'West Indies',
    'Afghanistan',
    'Pakistan',
    'Sri Lanka'    
]

delivery_df = delivery_df[delivery_df['batting_team'].isin(teams)]
delivery_df = delivery_df[delivery_df['bowling_team'].isin(teams)]

output = delivery_df[['match_id','batting_team','bowling_team','ball','runs','player_dismissed','city','venue']]

pickle.dump(output,open('data/processed/dataset_level2.pkl','wb'))