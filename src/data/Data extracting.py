# import numpy as np 
# import pandas as pd 
# import os 
# from tqdm import tqdm
# from yaml import safe_load
# import pickle

# # filenames = []
# # for file in os.listdir('data'):
# #     filenames.append(os.path.join('data',file))

# # dfs = []
# # counter = 1

# # for file in tqdm(filenames):
# #     with open(file, 'r') as f:
# #         df = pd.json_normalize(safe_load(f))
# #         df['match_id'] = counter
# #         dfs.append(df)
# #         counter += 1

# # final_df = pd.concat(dfs, ignore_index=True)

# # final_df.to_excel("data_t20.xlsx", index=False)

# df = pd.read_excel("data_t20.xlsx")

# df_final  = df.to_csv("data_t20.csv", index=False)

# # import pickle
# # pickle.dump(final_df , open('dataset_level_new.pkl','wb'))


