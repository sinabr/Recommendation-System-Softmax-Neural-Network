import scipy.sparse as sp
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader , Dataset
import random
from copy import deepcopy
random.seed(0)


""" Dataset Class from Pytorch """
class RatingData(Dataset):
    def __init__(self,user,item,target):
        self.user_tensor = user
        self.item_tensor = item
        self.target_tensor = target

    


class Data:
    def __init__ (self,file_name,columns,sep,feedback):
        # Load Data
        ratings = pd.read_csv(file_name,names=columns,sep=sep,engine='python')

        userIds = ratings[['oldId']].drop_duplicates()
        userIds['userId'] = np.arange(len(userIds))
        # Change User Id Range to [0, Number of Users - 1]
        ratings = pd.merge(ratings, userIds, on='oldId' ,how='left')

        movieIds = ratings[['movieId']].drop_duplicates()
        movieIds['itemId'] = np.arange(len(movieIds))
        # Change Item Id Range to [0, Number of Items - 1]
        ratings = pd.merge(ratings, movieIds , on='movieId',how='left')
       
        self.ratings = ratings

        print('Number of Movies : ')
        print(len(movieIds))
        print('Numebr of Users : ')
        print(len(userIds))

        """
        There are two ways to deal with ratings:

        Implicit Feedback : Binary Ratings

        Explicit Feedback : Normalized Ratings

        """

        self.all_users = set(self.ratings['userId'].unique())
        self.all_items = set(self.ratings['itemId'].unique())
        self.n_users = len(self.all_users)
        self.n_items = len(self.all_items)


        copy_ratings = deepcopy(ratings)
        
        # feedback = 'explicit'
        feedback = 'implicit'
        """ We Use deepcopy(df) To Not Modify the original dataframe """
        if feedback == 'implicit':
            max_val = ratings['rating'].max()
            ratings['rating'] = ratings['rating'] / max_val

        elif feedback == 'explicit':
            # rating equal to one if not zero -> Binary
            ratings['rating'][ratings['rating'] > 0] = 1.0

        self.ratings = ratings
        self.original = copy_ratings
        self.train_set, self.test_set = self.test_train_split(2, self.ratings)
        self.negative_items = self.negative_items(self.ratings,feedback=feedback,n_samples=2)


    def test_train_split(self, test_number, ratings):
        """ Add timestamp rank attribute, to be able to sort by timestamp """
        # ratings['time_rank'] = ratings.groupby(['userId'])['timestamp'].sort_values(axis=1,ascending=False)
        ratings['time_rank'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['time_rank'] <= test_number]
        train = ratings[ratings['time_rank'] > test_number]
        return train, test

    def negative_items(self, ratings, feedback='explicit', n_negatives=10):
        self.n_negatives = n_negatives
        """ Get the list of items a user interacted """
        # interaction_lists = ratings.groupby('userId')['itemId'].reindex(columns=['itemId':'interactions'])
        if feedback == 'explicit':
            # Ratings Below <negative> are considered negative
            negative = 3
            negative_ratings = ratings[ratings['ratings'] < negative]
            negative_by_users = negative_ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(columns={'itemId' == 'interacted'})
            negative_by_users['negative_items'] = negative_by_users['interacted'].apply(lambda items: self.all_items - items)
            # negative_by_users['negative_sampels'] = negative_by_users['negative_items'].apply(lambda items: random(items, n_negatives))
            return negative_by_users

        elif feedback == 'implicit':
            # In Implicit feedback, Not Interacted is Negative
            interactions = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(columns={'itemId':'interacted'})
            interactions['negative_items'] = interactions['interacted'].apply(lambda items: self.all_items - items)
            # samples from the negative items
            # interactions['negative_samples'] = interactions['negative_items'].apply(lambda items: random.sample(items, n_negatives))
            return interactions

    def construct_train_loader(self, n_negatives, batch_size):
        """ Training Data for an Epoch """
        users = []
        items = []
        ratings = []
        # train_data = pd.merge(self.train_set, self.negative_items[['userId','negative_items']], on='userId' )
        # train_data['negative_samples'] = train_data['negative_items'].apply(lambda items: random(items,))
        for sample in self.train_set.itertuples():
            users.append(int(sample.userId))
            items.append(int(sample.itemId))
            ratings.append(float(sample.rating))

        for sample in self.negative_items.itertuples():
            for i in range(self.n_negatives):
                users.appned(int(sample.userId))
                items.append(int(sample.itemId))
                ratings.append(float(sample.negative_items[i]))

        # Construct Tensors
        user_tensor = torch.LongTensor(users)
        item_tensor = torch.LongTensor(items)
        ratings_tensor = torch.LongTensor(ratings)

        dataset = RatingData(user=user_tensor,item=item_tensor,target=ratings_tensor)
        
        # Create Dataloader with Shuffled data in Batches
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloader

    def evaluation_set(self):
        eval_ratings = pd.merge(self.test_set, self.negatives[['userId', 'negative_samples']], on='userId')

        eval_users, eval_items, eval_neg_users, eval_neg_items = [], [], [], []

        for item in eval_ratings.itertuples():

            eval_users.append(int(item.userId))
            eval_items.append(int(item.itemId))

            for i in range(len(item.negative_samples)):

                eval_neg_users.append(int(item.userId))
                eval_neg_items.append(int(item.negative_samples[i]))

        userTensor = torch.LongTensor(eval_users)
        itemTensor = torch.LongTensor(eval_items)
        negUserTensor = torch.LongTensor(eval_neg_users)
        negItemTensor = torch.LongTensor(eval_neg_items)

        return [userTensor, itemTensor, negUserTensor, negItemTensor]
