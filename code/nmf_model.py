import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from .eval import Metrics


class NMFModel(nn.Module):

    def __init__(self, options):

        # Initialize Inherited Class
        super(NMFModel, self).__init__()

        self.options = options

        #  Use "Metrics" for evaluation purposes
        self.metric = Metrics(n=10)

        self.optimizer = self.select_optimizer(options)

        # Loss Function
        if options['feedback'] == 'explicit':
            # Binary Cross Entropy
            self.loss_func = nn.BCELoss()
        else:
            # Mean Square Error
            self.loss_func = nn.MSELoss()

        # Initialize Child Class
        self.n_users = options['n_users']
        self.n_items = options['n_items']

        # GMF, MLP Parameters
        self.mlp_dimension = self.options['mlp_dim']
        self.gmf_dimension = self.options['gmf_dim']

        # User and Item Embedding lookup tables in torch Embeddings for GMF
        self.gmf_user_embedding = nn.Embedding(embedding_dim=self.gmf_dimension, num_embeddings=self.n_users)
        self.gmf_item_embedding = nn.Embedding(embedding_dim=self.gmf_dimension, num_embeddings=self.n_items)

        # User and Item Embedding lookup tables in torch Embeddings for MLP
        self.mlp_user_embedding = nn.Embedding(embedding_dim=self.mlp_dimension, num_embeddings=self.n_users)
        self.mlp_item_embedding = nn.Embedding(embedding_dim=self.mlp_dimension, num_embeddings=self.n_items)

        # Layer Shapes:
        self.layers = [16, 64, 32, 16, 8]
        # self.layers = options.layers
        shapes = enumerate(zip(self.layers[:-1], self.layers[1:]))

        # A layer with input size of features and output of 1
        self.output = nn.Linear(in_features=self.layers[-1], out_features=1)
        self.activation = nn.Sigmoid()
        # self.activation = nn.Relu()

        # Layer List Module for the MLP
        self.layers = nn.ModuleList()

        # Create Linear Layers
        for _, n_in, n_out in shapes:
            self.layers.append(nn.Linear(n_in, n_out))

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
        gmf_user_embeddings = self.gmf_user_embedding(users)
        gmf_item_embeddings = self.gmf_item_embedding(items)

        mlp_user_embeddings = self.mlp_user_embedding(users)
        mlp_item_embeddings = self.mlp_item_embedding(items)

        """ Input: Concatenation of User and Item """
        input_ = torch.cat([mlp_user_embeddings, mlp_item_embeddings], dim=-1, out=None)
        n_layers = (len(self.layers))
        for i in range(n_layers):
            layer = self.layers[i]
            input_ = layer(input_)
            input_ = nn.ReLU()(input_)

        gmf_vector = torch.mul(gmf_user_embeddings, gmf_item_embeddings)

        # We Feed the Concat of GMF & MLP Result to the last layer:
        gmf_mlp = torch.cat([input_, gmf_vector], dim=-1, out=None)

        """ More Effiecient: """
        last_out = self.output(gmf_mlp)
        # Run the Sigmoid
        pred_rate = self.activation(last_out)
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

    # We Train the GMF and MLP and Load the trained values here
    def pretrain_load(self):
        pass
