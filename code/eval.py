import pandas as pd
import math


class Metrics:
    def __init__(self, n, test_data):
        """ Number of Recommended Items """
        self.test_set = None
        self.top_k = n

    def set_attrs(self, test_data):
        """
        test_data:
        [0]:test_ranks
        [1]:test_users
        [2]:test_items
        negatives:
        [3]:negative_ranks
        [4]:negative_users
        [5]:negative_items

        """
        test_ranks = test_data[0]
        test_users = test_data[1]
        test_items = test_data[2]
        negative_ranks = test_data[3]
        negative_users = test_data[4]
        negative_items = test_data[5]

        # test set -> only positive feedback
        test_pos = pd.DataFrame({'user': test_users, 'test_item': test_items, 'test_score': test_ranks})
        # test set -> positive and negative feedback
        test_with_negs = pd.DataFrame({'user': negative_users + test_users, 'item': negative_items + test_items,
                                       'score': negative_ranks + test_ranks})

        test_set = pd.merge(test_with_negs, test_pos, on=['user'], how='left')
        # rank the items according to the ranks in samples by user
        test_set['rank'] = test_set.groupby('user')['score'].rank(method='first', ascending=False)
        test_set.sort_values(['user', 'rank'], inplace=True)
        self.test_set = test_set

    def ndgc(self):
        set = self.test_set
        # pick the top k (10)
        top_k = set[set['rank'] <= self.top_k]
        #
        test_in_top_k = self.top_k[self.top_k['test_item'] == top_k['item']]
        test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(
            lambda x: math.log(2) / math.log(1 + x))  # the rank starts from 1
        return test_in_top_k['ndcg'].sum() * 1.0 / set['user'].nunique()

    def hit(self):
        set, k = self.test_set, self.top_k
        top_k = set[set['rank'] <= k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]  # golden items hit in the top_K items
        return len(test_in_top_k) * 1.0 / set['user'].nunique()
