#/usr/bin/python
from __future__ import print_function

import optuna
import argparse
import torch
import pickle 
import numpy as np 
import os 
import math 
import random 
import sys
import matplotlib.pyplot as plt 
import data
import scipy.io


from torch import nn, optim
from torch.nn import functional as F
from pathlib import Path
from gensim.models.fasttext import FastText as FT_gensim
import tracemalloc

from etm import ETM
from utils import nearest_neighbors, get_topic_coherence, get_topic_diversity

parser = argparse.ArgumentParser(description='The Embedded Topic Model')

### data and file related arguments
parser.add_argument('--dataset', type=str, default='min_df_100_tok_30', help='name of corpus')
parser.add_argument('--data_path', type=str, default='data', help='directory containing data')
parser.add_argument('--emb_path', type=str, default='data/min_df_100_tok_30/', help='directory containing word embeddings')
parser.add_argument('--save_path', type=str, default='./results', help='path to save results')
parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for training')
parser.add_argument('--topK', type=int, default=0, help='number of tokens to be replaced')

### model-related arguments
parser.add_argument('--num_topics', type=int, default=50, help='number of topics')
parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu', help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
parser.add_argument('--train_embeddings', type=int, default=0, help='whether to fix rho or train it')
parser.add_argument('--sim_coeff', type=float, default=25.0, help='Invariance regularization loss coefficient')
parser.add_argument('--std_coeff', type=float, default=25.0, help='Variance regularization loss coefficient')
parser.add_argument('--cov_coeff', type=float, default=1.0, help='Covariance regularization loss coefficient')

### optimization-related arguments
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this...')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train...150 for 20ng 100 for others')
parser.add_argument('--mode', type=str, default='train', help='train or eval model')
parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--seed', type=int, default=2019, help='random seed (default: 1)')
parser.add_argument('--enc_drop', type=float, default=0.2, help='dropout rate on encoder')
parser.add_argument('--clip', type=float, default=0.0, help='gradient clipping')
parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
parser.add_argument('--anneal_lr', type=int, default=0, help='whether to anneal the learning rate or not')
parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')

### evaluation, visualization, and logging-related arguments
parser.add_argument('--num_words', type=int, default=10, help='number of words for topic viz')
parser.add_argument('--log_interval', type=int, default=10, help='when to log training')
parser.add_argument('--visualize_every', type=int, default=10, help='when to visualize results')
parser.add_argument('--eval_batch_size', type=int, default=1000, help='input batch size for evaluation')
parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')
parser.add_argument('--tc', type=int, default=0, help='whether to compute topic coherence or not')
parser.add_argument('--td', type=int, default=0, help='whether to compute topic diversity or not')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('\n')
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

with open(args.data_path+'/'+args.dataset+'/vocab.pkl','rb') as f:
    vocab=pickle.load(f)
vocab_size = len(vocab)
args.vocab_size = vocab_size

with open(args.data_path+'/'+args.dataset+'/dataset_20ng.pkl','rb') as f:
    datasets = pickle.load(f)


def objective(trial):

    args.lr = trial.suggest_float('lr', 0.001, 0.01, step=0.001)
    args.enc_drop = trial.suggest_float('enc_drop', 0.1, 0.5, step=0.1)
    args.batch_size = trial.suggest_int('batch_size', 100, 1000, step=100)

    # 1. training data
    training_set = datasets['train']
    args.num_docs_train = training_set['tokens'].shape[0]
    corpus_training = data.get_batch(training_set, range(args.num_docs_train), vocab_size, device)

    # 1.1 substitute_min_indices for all the training samples
    if args.topK != 0:
        _, substitute_min_indices_all = data.get_batch(training_set, range(args.num_docs_train), vocab_size, device, topK=args.topK)
    else:
        substitute_min_indices_all = None

    # 2. dev set
    valid_set = datasets['validation']['val']
    args.num_docs_valid = valid_set['tokens'].shape[0]
    corpus_valid = data.get_batch(valid_set, range(args.num_docs_valid), vocab_size, device)

    # 3. test data
    test_set = datasets['test']['test']
    args.num_docs_test = test_set['tokens'].shape[0]
    corpus_test = data.get_batch(test_set, range(args.num_docs_test), vocab_size, device)


    embeddings = None
    if not args.train_embeddings:
        # embeddings = data.read_embedding_matrix(vocab, device, load_trainned=False)
        with open(args.emb_path + 'embed_matrix.pkl','rb') as f:
            embeddings =  pickle.load(f)
        embeddings = torch.tensor(embeddings)
        args.embeddings_dim = embeddings.size()

    print('=*'*100)
    print('Training an Embedded Topic Model on {} with the following settings: {}'.format(args.dataset.upper(), args))
    print('=*'*100)

    ## define checkpoint
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)



    if args.mode == 'eval':
        args.ckpt = args.load_from
    else:
        args.ckpt = Path.cwd().joinpath(args.save_path, 
            'etm_{}_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_trainEmbeddings_{}_topK_{}'.format(
            args.dataset, args.num_topics, args.t_hidden_size, args.optimizer, args.clip, args.theta_act, 
                args.lr, args.batch_size, args.rho_size, args.train_embeddings, args.topK))

    optuna_flag = 1
    ## define model and optimizer
    model = ETM(args.num_topics, 
                vocab_size, 
                args.t_hidden_size, 
                args.rho_size, 
                args.emb_size, 
                args.theta_act,
                args.sim_coeff,
                args.std_coeff,
                args.cov_coeff,
                args.ckpt, 
                optuna_flag,
                embeddings, 
                args.train_embeddings, 
                args.enc_drop).to(device)

    print('model: {}'.format(model))

    optimizer = model.get_optimizer(args)


    tracemalloc.start()
    if args.mode == 'train':
        ## train model on data 
        best_epoch = 0
        best_val_ppl = 1e9
        all_val_ppls = []
        print('\n')
        print('Visualizing model quality before training...', args.epochs)
        #model.visualize(args, vocabulary = vocab)
        print('\n')
        for epoch in range(0, args.epochs):
            print("I am training for epoch", epoch)
            model.train_for_epoch(epoch, args, training_set, substitute_min_indices_all = substitute_min_indices_all)
            # val_ppl = model.evaluate(args, 'val', training_set, vocab,  test_1, test_2)
            continue_training = model.validate(args, 'val', valid_set, training_set, vocab)
            if not continue_training:
                break 

        model = model.to(device)
        model.eval()
        neg_npmi_mean = model.evaluate(corpus_valid, vocab)
        
        ## calculate topic coherence and topic diversity
        # model.eval()
        # model.evaluate(test_set, vocab)
        return neg_npmi_mean
    else:   
        # print('do nothing.')
        with open(ckpt, 'rb') as f:
            model = torch.load(f)
        model = model.to(device)
        model.eval()
        ## calculate topic coherence and topic diversity
        model.evaluate(test_set, vocab)


    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()

TRIAL_SIZE = 100
study = optuna.create_study()
study.optimize(objective, n_trials=TRIAL_SIZE)

# print best hyperparameters
print('Best Hyperparameters: ', study.best_params)