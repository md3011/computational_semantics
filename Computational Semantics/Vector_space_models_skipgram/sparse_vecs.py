#!/usr/bin/env python3
""" sparse_vecs.py

    Run this script (use Python 3!) with the --help flag to see how to use command-line options.

    Hint: Anything already imported for you might be useful, but not necessarily required, to use :)
"""

import argparse
import collections

import numpy as np
from scipy.spatial.distance import pdist, squareform
from operator import itemgetter

import utils

""" hyperparameters ( do not modify! ) """
WINDOW_SIZE = 2
NUM_CLOSEST = 20

idtoindex = {}
inverse_vocab_truncated = {}
def word_vectors(corpus, vocab):
    """
        Input: A corpus (list of list of string) and a vocab (word-to-id mapping)
        Output: A lookup table that maps [word id] -> [word vector]
    """

    visited_words = {}
    word_vecs = {}

    for sentence in corpus:
        for i in range(len(sentence)):
            current_word = sentence[i]

            if visited_words.get(vocab[current_word],"n") == "n":
                word_vecs[vocab[current_word]] = np.zeros(len(vocab))
                visited_words[vocab[current_word]] = "y"

            for neighbor in range((i-WINDOW_SIZE),(i+WINDOW_SIZE+1)):
                if (neighbor >= 0) and (neighbor < len(sentence)) and (neighbor != i):
                    current_array = word_vecs[vocab[current_word]]
                    current_array[[idtoindex[vocab[sentence[neighbor]]]]] += 1.0
                    word_vecs[vocab[current_word]] = current_array

    return word_vecs
#    raise NotImplementedError("word_vectors")



def most_similar(lookup_table, wordvec):
    """ Helper function (optional).

        Given a lookup table and word vector, find the top most-similar word ids to the given
        word vector. You can limit this to the first NUM_CLOSEST results.
    """

    #distances dictionary of the form id:distance
    distances = {}

    for word_id in lookup_table:
        distances[word_id] = np.sum((lookup_table[word_id] - wordvec)**2)

    sorted_dict = sorted(distances.items(), key=itemgetter(1))[1:NUM_CLOSEST+1]

    print (sorted_dict)

    for each in sorted_dict:
        if each[1]==0:
            continue
        else:
            print(inverse_vocab_truncated[each[0]])

    # raise NotImplementedError("most_similar")

def main():
    """
    Task: Transform a corpus of text into word vectors according to this context-window principle.
    1. Load the data - this is done for you.
    2. Construct a vocabulary across the entire corpus. This should map a string (word) to id.
    3. Use the vocabulary (as a word-to-id mapping) and corpus to construct the sparse word vectors.
    """
    sentences = utils.load_corpus(args.corpus)
    # print (len(sentences))

    vocab_full, inverse_vocab_full = utils.construct_vocab(sentences)
    # print (vocab_full)
    # print (inverse_vocab_full)

    counts = utils.word_counts(sentences)
    new_corpus, new_counts = utils.trunc_vocab(sentences, counts)

    # print (len(new_corpus))
    # print ("**********************************************")
    # print (new_counts)

    global inverse_vocab_truncated
    vocab_truncated = {}
    inverse_vocab_truncated = {}

    for word in new_counts:
        vocab_truncated[word] = vocab_full[word]
        inverse_vocab_truncated[vocab_full[word]] = word

    print (vocab_truncated)
    # print (inverse_vocab_truncated)
    favorite = input("Enter Favorite Word\n")
    global idtoindex
    idtoindex = {}

    i = 0

    for word_id in inverse_vocab_truncated:
        idtoindex[word_id] = i
        i += 1

    # print (idtoindex)
    lookup_table = word_vectors(new_corpus, vocab_truncated)

    most_similar(lookup_table, lookup_table[vocab_truncated[favorite]])
    # print (lookup_table)
    """ TODO: Implement what you need to answer the writeup questions. """


if __name__ == "__main__":
    # NOTE: feel free to add your own arguments here, but we will only ever run your script
    #  based on the arguments provided in this stencil
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, help="path to corpus file", default="corpus.txt")
    args = parser.parse_args()

    np.set_printoptions(linewidth=150)

    main()
