""" This file defines modules for looking up embeddings given word ids. """

from os import path
import pickle

import numpy as np
import torch
from torch import nn

import allennlp.modules.elmo as allen_elmo

# These are the different embedding sizes. Feel free to experiment
# with different sizes for random.
sizes = {"elmo": 1024, "glove": 200, "random": 10}
sizes["both"] = sizes["elmo"] + sizes["glove"]

class Elmo(nn.Module):
    """ TODO: Finish implementing __init__, forward, and _get_charids for Elmo embeddings.
        Take a look at the Allen AI documentation on using Elmo:
            https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md
        In particular, reference the section "Using ELMo as a PyTorch Module".
        In addition, the Elmo model documentation may be useful:
            https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py#L34
    """

    def __init__(self, idx2word, device=torch.device('cpu')):
        """ Load the ELMo model. The first time you run this, it will download a pretrained model. """
        super(Elmo, self).__init__()
        options = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weights = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        #raise NotImplementedError("Elmo.__init__")
        self.elmo = allen_elmo.Elmo(options, weights, 2, dropout=0) # TODO initialise an allen_elmo.Elmo model
        self.idx2word = idx2word # Note: you'll need this mapping for _get_charids

        self.embed_size = sizes["elmo"]
        self._dev = device
        self.embeddings = None
    def forward(self, batch):
        char_ids = self._get_charids(batch)
        # TODO get elmo embeddings given char_ids:
        self.embeddings = self.elmo(char_ids)
        return self.embeddings['elmo_representations'][0]
        #raise NotImplementedError("Elmo.forward")

    def _get_charids(self, batch):
        """ Given a batch of sentences, return a torch tensor of character ids.
                :param batch: List of sentences - each sentence is a list of int ids
            Return:
                torch tensor on self._dev
        """
        # 1. Map each sentence in batch to a list of string tokens (hint: use idx2word)
        # 2. Use allen_elmo.batch_to_ids to convert sentences to character ids.
        # raise NotImplementedError("Elmo._get_charids")
        to_return = []
        #print(batch)
        for sentence in batch:
            templ = []
            for ids in sentence:
                templ.append(self.idx2word[ids])
            to_return.append(templ)
        #print (to_return)
        to_return = allen_elmo.batch_to_ids(to_return)
        return torch.tensor(to_return).to(self._dev)

class Glove(nn.Module):
    def __init__(self, data_dir, idx2word, device=torch.device('cpu')):
        """ TODO load pre-trained GloVe embeddings from disk """
        super(Glove, self).__init__()
        # 1. Load glove.6B.200d.npy from inside data_dir into a numpy array
        #    (hint: np.load)
        # 2. Load glove_tok2id.dict from inside data_dir. This is used to map
        #    a word token (as str) to glove vocab id (hint: pickle.load)
        # raise NotImplementedError("Glove.__init__")

        # self.pathfull = path.join(data_dir,"glove.6B.200d.npy")        
        self.glove_pretrained = np.load(path.join(data_dir,"glove.6B.200d.npy"))

        # self.pathfull2 = path.join(data_dir,"glove_tok2id.dict")
        with open(path.join(data_dir,"glove_tok2id.dict"),"rb") as openfile:
            self.glove_vocab = pickle.load(openfile)

        self.idx2word = idx2word
        self.embed_size = sizes["glove"]

        # self.glove_vocab = {}
        # for each in self.glove_tok2id:
        #     self.glove_vocab[each] = self.glove_pretrained[self.glove_tok2id[each]]

        # 3. Create a torch tensor of the glove vectors and construct a
        #    a nn.Embedding out of it (hint: see how RandEmbed does it)
        # self.embeddings = None # nn.Embedding layer
        self.embeddings = nn.Embedding.from_pretrained(torch.from_numpy(self.glove_pretrained))
        # print(self.embeddings)

        self._dev = device

    def _lookup_glove(self, word_id):
        # given a word_id, convert to string and get glove id from the string:
        # unk if necessary.
        return self.glove_vocab.get(self.idx2word[word_id].lower(), self.glove_vocab["unk"])

    def _get_gloveids(self, batch):
        # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.apply_
        # print(batch)
        return batch.apply_(self._lookup_glove).to(self._dev)

    def forward(self, batch):
        return self.embeddings(self._get_gloveids(batch)).to(self._dev)

class ElmoGlove(nn.Module):
    def __init__(self, data_dir, idx2word, device=torch.device('cpu')):
        """ TODO construct Elmo and Glove lookup instances """
        super(ElmoGlove, self).__init__()

        #raise NotImplementedError("ElmoGlove.__init__")
        self.elmoobj = Elmo(idx2word,device)
        self.gloveobj = Glove(data_dir,idx2word,device)
        self.elmo = None # TODO
        self.glove = None # TODO

        self.embed_size = sizes["both"]
        self._dev = device

    def forward(self, batch):
        """ TODO Concatenate ELMo and GloVe embeddings together """
        # raise NotImplementedError("ElmoGlove.forward")
        self.elmo = self.elmoobj(batch)
        self.glove = self.gloveobj(batch)

        return torch.cat((self.elmo,self.glove),dim=2).to(self._dev)

class RandEmbed(nn.Module):
    def __init__(self, vocab_size, device=torch.device('cpu')):
        super(RandEmbed, self).__init__()
        self.embed_size = sizes["random"]
        self._dev = device

        self.embeddings = nn.Embedding.from_pretrained(torch.rand(vocab_size, self.embed_size))

    def forward(self, batch):
        batch = batch.to(self._dev)
        return self.embeddings(batch)

