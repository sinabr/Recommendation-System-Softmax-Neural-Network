import scipy.sparse as sp
import numpy as np
import pandas as pd

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
       
        print('Number of Movies : ')
        print(len(movieIds))
        print('Numebr of Users : ')
        print(len(userIds))

        """
        There are two ways to deal with ratings:

        Implicit Feedback : Binary Ratings

        Explicit Feedback : Normalized Ratings

        """

        copy_ratings = deepcopy(ratings)
        
        # feedback = 'explicit'
        feedback = 'implicit'
        """ We Use deepcopy(df) To Not Modify the original dataframe """
        if feedback == 'implicit':
            max_val = ratings['rating'].max()
            ratings['rating'] = ratings['rating'] / max_val

        elif feedback == 'explicit':
            # rating equal to one if not zero
            ratings['rating'][ratings['rating'] > 0] = 1.0

        self.ratings = ratings
        self.original = copy_ratings
        
    def test_train_split(self):
        