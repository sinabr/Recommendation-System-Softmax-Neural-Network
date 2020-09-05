import torch.nn as nn
import pandas as pd
import torch
import torch.optim as optim


class GMFModel(nn.Module):
    def __init__(self,options):
        # Initialize Inherited Class
        super(GMFModel,self).__init__()

        #  Use MetronAtk for evaluation purposes
        self.metric = MetronAtk(top_k= 10)

        self.optimizer = select_optimizer(self, options.optimization)
            

        self.dimension = options['embedding_dimension']


        # Initialize Child Class
        self.n_users = options['n_users']
        self.n_items = options['n_items']


        # User and Item Embedding lookup tables in torch Embeddings
        self.user_embedding = nn.Embedding(embedding_dim=self.dimension, num_embeddings=self.n_users)
        self.item_embedding = nn.Embedding(embedding_dim=self.dimension, num_embeddings=self.n_items)

        # A layer with input size of features and output of 1
        self.output = nn.Linear(in_features= self.dimension, out_features=1)
        self.activation = nn.Sigmoid()
        #self.activation = nn.Relu()

    def select_optimizer(self, options):
        if options['algorithm'] == 'adagrad' :
            return optim.Adagrad(self.parameters(),lr=options['learning_rate'],weight_decay=options['regularization'])
        elif options['algorithm'] == 'sgd' :
            return optim.SGD(self.parameters(),lr=options['learning_rate'],momentum=options['momentum'],weight_decay=options['regularization'])
        elif options['algorithm'] == 'adam' :
            return optim.Adam(self.parameters(),lr=options['learning_rate'],weight_decay=options['regularization'])
        elif options['algorithm'] == 'spare_adam' :
            return optim.SparseAdam()
        elif options['algorithm'] == 'rms_prop':
            return optim.RMSprop(self.parameters(),lr=options['learning_rate'],momentum=options['momentum'],alpha=options['alpha'])


    def feed_forward(self,user,item):
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        dot_product = torch.mul(user_embedding, item_embedding)
        pred_rate = self.activation(self.output(dot_product))
        return pred_rate

    def train_batch(self, users, items, ratings):
