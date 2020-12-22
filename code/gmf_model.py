import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from .eval import Metrics


class GMFModel(nn.Module):
    def __init__(self, options):
        # Initialize Inherited Class
        super(GMFModel, self).__init__()

        #  Use Metrics for evaluation purposes
        self.metric = Metrics(top_k=10)

        self.optimizer = self.select_optimizer(options)

        # Loss Function
        if options['feedback'] == 'explicit':
            # Binary Cross Entropy
            self.loss_func = nn.BCELoss()
        else:
            # Mean Square Error
            self.loss_func = nn.MSELoss()

        self.dimension = options['embedding_dimension']

        # Initialize Child Class
        self.n_users = options['n_users']
        self.n_items = options['n_items']

        # User and Item Embedding lookup tables in torch Embeddings
        self.user_embedding = nn.Embedding(embedding_dim=self.dimension, num_embeddings=self.n_users)
        self.item_embedding = nn.Embedding(embedding_dim=self.dimension, num_embeddings=self.n_items)

        # A layer with input size of features and output of 1
        self.output = nn.Linear(in_features=self.dimension, out_features=1)
        self.activation = nn.Sigmoid()
        # self.activation = nn.Relu()

    def select_optimizer(self, options):
        if options['algorithm'] == 'adagrad':
            return optim.Adagrad(self.parameters(), lr=options['learning_rate'], weight_decay=options['regularization'])
        elif options['algorithm'] == 'sgd':
            return optim.SGD(self.parameters(), lr=options['learning_rate'], momentum=options['momentum'],
                             weight_decay=options['regularization'])
        elif options['algorithm'] == 'adam':
            return optim.Adam(self.parameters(), lr=options['learning_rate'], weight_decay=options['regularization'])
        elif options['algorithm'] == 'spare_adam':
            return optim.SparseAdam(self.parameters(), lr=options['learning_rate'])
        elif options['algorithm'] == 'rms_prop':
            return optim.RMSprop(self.parameters(), lr=options['learning_rate'], momentum=options['momentum'],
                                 alpha=options['alpha'])

    def feed_forward(self, users, items):
        user_embeddings = self.user_embedding(users)
        item_embeddings = self.item_embedding(items)

        """ Dot Product : """
        # dot_product = user_embeddings * item_embeddings

        """ More Effiecient : """
        dot_product = torch.mul(user_embeddings, item_embeddings)
        pred_rate = self.activation(self.output(dot_product))
        return pred_rate

    def train_batch(self, users, items, ratings):
        self.optimizer.zero_grad()
        # Reshape The Tensor
        predictions = self.feed_forward(users, items).view(-1)
        # Compute Loss
        batch_loss = self.loss_func(predictions, ratings)
        # Back Propagation
        batch_loss.backward()
        # Parameter Update Based On The Gradient
        self.optimizer.step()
        # loss value as a Float
        loss = batch_loss.item()
        return loss

    def train_epoch(self, train_data):
        self.train()
        loss = 0
        for batch_number, batch in enumerate(train_data):
            user = batch[0]
            item = batch[1]
            # rate = float(batch[2])
            rate = batch[2].float()
            batch_loss = self.train_batch(user, item, rate)
            loss = loss + batch_loss

        print('Loss : ', loss)
        return

    def run_evaluation(self, e_id, test_set):
        # Notify all your layers that you are in eval mode
        self.eval()
        # Disable gradient calculations
        with torch.no_grad:
            test_items = test_set[1]
            test_users = test_set[0]

            neg_items = test_set[3]
            neg_users = test_set[2]

            test_scores = self.feed_forward(test_users, test_items)
            neg_scores = self.feed_forward(neg_users, neg_items)

            eval_attributes = [
                test_scores,
                test_users,
                test_items,
                neg_scores,
                neg_users,
                neg_items
            ]

            self.metric.set_attrs(eval_attributes)

            hit_score = self.metric.hit()
            ndgc_score = self.metric.ndgc()

            print('EPOCH : ' + e_id + ' Hit Ratio : ' + hit_score + ' NDGC Score : ' + ndgc_score)

            return hit_score, ndgc_score

