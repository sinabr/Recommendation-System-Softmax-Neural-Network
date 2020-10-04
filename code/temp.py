
import pandas as pd
import numpy as np
from copy import deepcopy
import random

ratings = pd.read_csv('data/datasets/ml-1m/ratings.dat', sep='::', names=['oldId','movieId','rating','timestamp'] , engine='python')

# userIds = ratings['userId'].unique()
# print(ratings['userId'].isnull().any())

"""
Drop Duplicates : drop duplicate values and return Dataframe

Note : .unique() returns array

Here We Reindex userId and itemId in the Dataframe 

"""
# print(ratings['userId'].head())


userIds = ratings[['oldId']].drop_duplicates()
userIds['userId'] = np.arange(len(userIds))

ratings = pd.merge(ratings, userIds, on='oldId' ,how='left')

movieIds = ratings[['movieId']].drop_duplicates()
movieIds['itemId'] = np.arange(len(movieIds))

ratings = pd.merge(ratings, movieIds , on='movieId',how='left')

print('Number of Movies : ')
print(len(movieIds))
print('Numebr of Users : ')
print(len(userIds))

"""
There are two ways to deal with ratings:

Implicit Feedback : Binary Ratings

Explicit Feedback : Normalized Ratings

"""

# feedback = 'explicit'
feedback = 'implicit'
""" We Use deepcopy(df) To Not Modify the original dataframe """
# if feedback == 'implicit':
    
#     copy_ratings = deepcopy(ratings)
#     max_val = ratings['rating'].max()
#     ratings['rating'] = ratings['rating'] / max_val

# elif feedback == 'explicit':
    
#     copy_ratings = deepcopy(ratings)
#     # rating equal to one if not zero
#     ratings['rating'][ratings['rating'] > 0] = 1.0
ratings = ratings[ratings['itemId'] < 3]
ratings = ratings[ratings['userId'] < 4]
ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
print(ratings.head(30))
test = ratings[ratings['rank_latest'] == 1]
train = ratings[ratings['rank_latest'] > 1]
print(test.head(20))
print(train.head(20))

# item_pool = set(ratings['userId'].unique())


# # interact_status = ratings.groupby('userId')['itemId'].apply(set).reindex(['interactions'],axis='columns')
# interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(columns={'itemId':'interactions'})


# interact_status['negative_items'] = interact_status['interactions'].apply(lambda x: item_pool - x)


# print(interact_status.head())

# interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 99))
# print(interact_status[['userId', 'negative_items', 'negative_samples']].head(1))