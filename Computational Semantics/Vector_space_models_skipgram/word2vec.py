#!/usr/bin/env python3

import os
import time
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from scipy.spatial.distance import cosine
from operator import itemgetter

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from tqdm import tqdm

import utils
import hyperparams as hp

class SkipGramNetwork(nn.Module):

    def __init__(self, vocab_size, embedding_size):
        super(SkipGramNetwork, self).__init__()
        self.embeddings = nn.Linear(vocab_size, embedding_size)
        self.mid = nn.Linear(embedding_size, vocab_size)
        self.onehot_lookup = torch.eye(vocab_size, vocab_size)
        
        
#        raise NotImplementedError("SkipGramNetwork.__init__")

    def forward(self, inputs):
        
        current_input = self.onehot_lookup[inputs]
        h = F.relu(self.embeddings(current_input))
        logits = self.mid(h)
        return F.log_softmax(logits, dim=1)
    
#        raise NotImplementedError("SkipGramNetwork.forward")

def skip_grams(corpus, vocab):
   
    skipgram_list = []
    WINDOW_SIZE = hp.WINDOW_SIZE
    
    for sentence in corpus:
        for i in range(len(sentence)):
            current_word = sentence[i]
#            if visited_words.get(vocab[current_word],"n") == "n":
#                word_vecs[vocab[current_word]] = np.zeros(len(vocab))
#                visited_words[vocab[current_word]] = "y"
            for neighbor in range((i-WINDOW_SIZE),(i+WINDOW_SIZE+1)):
                if (neighbor >= 0) and (neighbor < len(sentence)) and (neighbor != i):
                    skipgram_list.append((vocab[current_word],vocab[sentence[neighbor]]))
                    
    return skipgram_list

#    raise NotImplementedError("skip_grams")

def train(model, dataloader):
    
    loss_function = nn.NLLLoss() # optionally, you can use nn.CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=hp.LEARN_RATE)
    loss_history = [] # NOTE: use this
    for epoch in range(hp.NUM_EPOCHS):
        print("---- Epoch {} of {} ----".format(epoch+1, hp.NUM_EPOCHS))
        loss_per_epoch = 0
        num_batches = 0
        for batch_idx, (target, context) in enumerate(dataloader):
            target = torch.LongTensor(target).to(device)
            context = torch.LongTensor(context).to(device)

            model.zero_grad() # clear gradients (torch will accumulate them)
         
            probs = model(target) # forward pass: we can call model like a function
            loss = loss_function(probs, context)
            loss.backward() # back-propagate i.e. compute gradients
            optimizer.step() # perform the parameter update
            loss_history.append(loss.item())
        
#            raise NotImplementedError("training loop")

            if (batch_idx % 500) == 0:
                print("\t Batch {}".format(batch_idx))

        ckpt = os.path.join(args.save, "model-epoch{}.ckpt".format(epoch))
        torch.save(model.state_dict(), ckpt)
        print("Checkpoint saved to {}".format(ckpt))

    return loss_history

def most_similar(lookup_table, wordvec):
  
#    raise NotImplementedError("most_similar")
    distances = {}
    for word_id in range(len(lookup_table)):
        distances[word_id] = cosine(wordvec, lookup_table[word_id])

    sorted_dict = sorted(distances.items(), key=itemgetter(1))[:hp.NUM_CLOSEST+1]

    return sorted_dict

def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
                label,
                xy=(x, y),
                xytext=(5, 2),
                textcoords='offset points',
                ha='right',
                va='bottom')

    plt.savefig(filename)

  # pylint: disable=g-import-not-at-
  
#    print (sorted_dict)

#    for each in sorted_dict:
#        if each[1]==0:
#            continue
#        else:
#            print(inverse_vocab_truncated[each[0]])

def main():

    net = SkipGramNetwork(hp.VOCAB_SIZE, hp.EMBED_SIZE).to(device)
    print(net)

    if args.restore:
        net.load_state_dict(torch.load(args.restore))
        vocab, inverse_vocab = utils.load_data(args.restore)
        print("Model restored from disk.")
    else:
        sentences = utils.load_corpus(args.corpus)
        word_freqs = utils.word_counts(sentences)
        sentences, word_freqs = utils.trunc_vocab(sentences, word_freqs) # TODO
        sentences = utils.subsample(sentences, word_freqs)

        vocab, inverse_vocab = utils.construct_vocab(sentences) # TODO
        skipgrams = skip_grams(sentences, vocab) # TODO
        utils.save_data(args.save, vocab, inverse_vocab)

        loader = DataLoader(skipgrams, batch_size=hp.BATCH_SIZE, shuffle=True)
        loss_hist = train(net, loader) # TODO returns loss function evaluations as python list

        """ You can plot loss_hist for your writeup:
            plt.plot(loss_hist)
            plt.show()
        """
        plt.plot(loss_hist)
        plt.show()

    # the weights of the embedding matrix are the lookup table
    lookup_table = net.embeddings.weight.data.cpu().numpy()

    """ TODO: Implement what you need in order to answer the writeup questions. """

    nearest = most_similar(lookup_table, lookup_table[vocab['journeyed']])
    nearest_words = [inverse_vocab[w] for w in nearest if w in inverse_vocab]
    print('Nearest to {0}: {1}'.format('journeyed', nearest_words))
    
#    print('Dimension Reduction and Plotting')
#    reduced = TSNE().fit_transform(lookup_table)
#    plt.scatter(reduced[:,0], reduced[:,1])
#    plt.show()
    
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 500
    low_dim_embs = tsne.fit_transform(lookup_table[:plot_only, :])
    labels = [inverse_vocab[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels, 'tsne.png')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=str, help="path to corpus file")
    parser.add_argument("--device", help="pass --device cuda to run on gpu", default="cpu")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--save", help="path to save directory", default="saved_runs")
    group.add_argument("--restore", help="path to model checkpoint")
    args = parser.parse_args()

    if not os.path.exists(args.save):
        os.mkdir(args.save)

    np.set_printoptions(linewidth=150)
    device = torch.device('cpu')
    if 'cuda' in args.device and torch.cuda.is_available():
        device = torch.device(args.device)
    print("Using device {}".format(device))

    main()
