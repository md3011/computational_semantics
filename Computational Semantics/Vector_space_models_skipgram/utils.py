import os

import collections
from tqdm import tqdm
import numpy as np
import pickle
from operator import itemgetter

import hyperparams as hp

def construct_vocab(corpus):
    """
        Input: A list of list of string. Each string represents a word token.
        Output: A tuple of dicts: (vocab, inverse_vocab)
                vocab: A dict mapping str -> int. This will be your vocabulary.
                inverse_vocab: Inverse mapping int -> str
    """
    vocab = {}
    inverse_vocab = {}

    mapping = 0

    for sentence in corpus:
        for word in sentence:
            if vocab.get(word,"n") == "n":
                vocab[word] = mapping
                inverse_vocab[mapping] = word
                mapping += 1

    toreturn = (vocab,inverse_vocab)

    return toreturn

#    raise NotImplementedError("construct_vocab")

def trunc_vocab(corpus, counts):
    """ Limit the vocabulary to the 10k most-frequent words. Remove rare words from
         the original corpus.
        Input: A list of list of string. Each string represents a word token.
        Output: A tuple (new_corpus, new_counts)
                new_corpus: A corpus (list of list of string) with only the 10k most-frequent words
                new_counts: Counts of the 10k most-frequent words
        Hint: Sort the keys of counts by their values
    """

    sorted_dict = sorted(counts.items(), key=itemgetter(1), reverse=True)[:10000]

    new_counts = {}
    for word in sorted_dict:
        if new_counts.get(word[0], "n") == "n":
            new_counts[word[0]] = word[1]

    # print (new_counts)
    new_corpus = []

    for sentence in corpus:
        new_sentence = []
        for word in sentence:
            if(new_counts.get(word, "n") != "n"):
                new_sentence.append(word)

        if len(new_sentence) > 0:
            new_corpus.append(new_sentence)

    return (new_corpus,new_counts)

#    raise NotImplementedError("trunc_vocab")

def load_corpus(path):
    """ Reads the data from disk.
        Returns a list of sentences, where each sentence is split into a list of word tokens
    """
    with open(path, "r") as f:
        c = [line.split() for line in f]
    return c

def word_counts(corpus):
    """ Given a corpus (such as returned by load_corpus), return a dictionary
        of word frequencies. Maps string token to integer count.
    """
    return collections.Counter(w for s in corpus for w in s)

def keep_prob(word_prob):
    """
        Probability of keeping a word, as a function of that word's probability
        in the corpus: word_prob = word_freq/total_words_in_corpus
    """
    return (np.sqrt(word_prob/hp.T_SAMPLE) + 1)*(hp.T_SAMPLE/word_prob)

def subsample(corpus, freqs):
    """
        Implements subsampling of frequent words (Mikolov et al., 2013), as they tend to carry less
        information value than rare words.
          Input: a corpus (as returned by load_corpus) and a dictionary of normalised word frequencies
          Output: the corpus with words dropped (in-place) according to keep_prob
    """
    N = sum(f for f in freqs.values())
    for ix,sentence in tqdm(enumerate(corpus), desc="Subsampling"):
        corpus[ix] = list(filter(lambda w: np.random.rand() < keep_prob(freqs[w]/N), sentence))
    return corpus

def load_data(restore_path):
    dirname = os.path.dirname(restore_path)
    with open(os.path.join(dirname, "data.pkl"), "rb") as f:
        vocab, inverse_vocab = pickle.load(f)
        return vocab, inverse_vocab

def save_data(save_path, vocab, inverse_vocab):
    with open(os.path.join(save_path, "data.pkl"), "wb") as f:
        pickle.dump((vocab, inverse_vocab), f)
