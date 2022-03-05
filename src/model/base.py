import os
import logging
import pickle
import copy
from turtle import update

import numpy as np
from tqdm import tqdm

import gensim
import torch
import torch.nn.functional as F
import torch.optim as optim
from .lm import MWMLNetLMFineGrind
from ..data import Dictionary

logger = logging.getLogger()


class DocReader:
    def __init__(self,args_dict,chars_dict,features_dict,state_dict=None):
        self.chars_dict = chars_dict
        self.features_dict = features_dict
        self.bmes_dict = Dictionary(['B','M','E','S'])
        self.config = args_dict
        self.config["vocab_size"] = len(chars_dict)
        self.config["features_size"] = len(features_dict)
        self.config["n_class"] = 4
        self.config["type_class"] = 20
        self.config["chars_max_length"] = args_dict["chars_max_length"]
        # Building network. If normalize if false, scores are not normalized
        # 0-1 per paragraph (no softmax).
        if self.config["model_type"].lower() == 'mwmlnet':
            self.network = MWMLNetLMFineGrind(**self.config)
            print("MWMLNetLMFineGrind")
        else:
            raise RuntimeError('Unsupported model: %s' % self.config["model_type"])
        self.config["updates"] = 0
        # Load saved state
        if state_dict:
            # Load buffer separately
            self.network.load_state_dict(state_dict)
    def init_optimizer(self,optimizer=None):
        if self.config["fix_embeddings"]:
            for p in self.network.parameters():
                p.requires_grad = False
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            if self.config["optim_method"].lower() == 'sgd':
                self.optimizer = optim.SGD(parameters, lr=self.config["learning_rate"],
                                       momentum=self.config["momentum"],
                                       weight_decay=self.config["weight_decay"])
            elif self.config["optim_method"].lower() == 'adamax':
                self.optimizer = optim.Adamax(parameters,lr=self.config["learning_rate"],
                                          weight_decay=self.config["weight_decay"])
            elif self.config["optim_method"].lower() == 'adam':
                self.optimizer = optim.Adam(parameters,lr=self.config["learning_rate"],
                                          weight_decay=self.config["weight_decay"])
            elif self.config["optim_method"].lower() == 'adadelta':
                self.optimizer = optim.Adadelta(parameters,lr=self.config["learning_rate"],
                                            rho=self.config["rho"],eps=self.config["eps"],
                                            weight_decay=self.config["weight_decay"])
            else:
                raise RuntimeError('Unsupported optimizer: %s' %self.config["optim_method"])

    def save(self,save_file_name,epoch=0):
        state_dict = copy.copy(self.network.cpu().state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'char_dict': self.chars_dict,
            'feature_dict': self.features_dict,
            'config':self.config
        }
        try:
            torch.save(params, save_file_name)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')
    @staticmethod
    def load(loaded_file_name):
        logger.info('Loading model from file %s'%loaded_file_name)
        saved_params = torch.load(loaded_file_name,map_location=lambda storage,loc:storage)
        chars_dict = saved_params["chars_dict"]
        features_dict = saved_params["features_dict"]
        state_dict = saved_params["state_dict"]
        config = saved_params["config"]
        model = DocReader(config,chars_dict,features_dict,state_dict)
        model.init_optimizer()
        return model
    def checkpoint(self,save_file_name,epoch):
        params = {
            'state_dict':self.network.cpu().state_dict(),
            'chars_dict':self.chars_dict,
            'features_dict':self.features_dict,
            'config':self.config,
            'optimizer':self.optimizer，
            'epoch':epoch
        }
        try:
            torch.save(params,save_file_name)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')
    @staticmethod
    def load_checkpoint(checkpoint_file_name):
        logger.info('Loading model %s'%checkpoint_file_name)
        saved_params = torch.load(
            checkpoint_file_name,map_location=lambda storage,loc:storage
        )
        chars_dict = saved_params['chars_dict']
        features_dict = saved_params['features_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        optimizer = saved_params['optimizer']
        config = saved_params['config']
        model = DocReader(config, chars_dict, features_dict, state_dict)
        model.init_optimizer(optimizer)
        return model, epoch
    def load_embeddings(self,embedding_file,save_embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        if os.path.exists(save_embedding_file):
            with open(save_embedding_file,'rb') as rfp:
                data_dict = pickle.load(rfp)
            self.chars_dict = data_dict["chars_dict"]
            embedding = data_dict["embedding"]
            logger.info('Loading pre-trained embeddings for %d words from %s' %(len(self.chars_dict), save_embedding_file))
        else:
            logger.info('Loading pre-trained embeddings for %d words from %s' %(len(self.chars_dict), embedding_file))
            # When normalized, some words are duplicated. (Average the embeddings).
            vec_counts = {}

            model = gensim.models.KeyedVectors.load_word2vec_format(embedding_file)
            loaded_dim = model.vector_size
            words_list = model.index_to_key
            length = len(words_list)
            vocab_size = len(self.chars_dict)
            embedding = np.random.rand(vocab_size,loaded_dim)
            for index in tqdm(range(length),desc='Loading vectors'):
                word = words_list[index]
                if word in self.chars_dict:
                    vec = model.get_vector(word)
                    if word not in vec_counts:
                        vec_counts[word] = 1
                        embedding[self.chars_dict[word]] = vec
                    else:
                        logging.warning('WARN: Duplicate embedding found for %s' % word)
                        vec_counts[word] = vec_counts[word] + 1
                        embedding[self.chars_dict[word]] += vec
            del model
            for w, c in vec_counts.items():
                embedding[self.chars_dict[w]]/=c
            with open(save_embedding_file,'wb') as wfp:
                data_dict = {
                    "embedding":embedding,
                    "chars_dict":self.chars_dict
                }
                pickle.dump(data_dict,wfp)
        embedding = torch.tensor(embedding,dtype=torch.float)
        self.network.from_pretrained(embedding)
        logger.info('Loaded %d embeddings dimension(%d)' %(embedding.shape[0],embedding.shape[1]))
    def expand_dictionary(self,external_dict=None):
        # Add words to dictionary and expand embedding layer
        if external_dict is not None and len(external_dict)>0:
            logger.info('Adding %d new words to dictionary...' % len(external_dict))
            for w in external_dict:
                self.chars_dict.add(w)
            self.vocabs_size = len(self.chars_dict)
            logger.info('New vocab size: %d' % len(self.chars_dict))
    def to_device(self,device):
        self.network.to(device)
    def update(self,ex,device):
        """
            Forward a batch of examples; step the optimizer to update weights.
        """
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')
        # Train mode
        self.network.train()
        self.network.to(device)
        # Transfer to GPU
        # ex : words_segs,chars_segs,features_segs,labels
        inputs = [e.to(device) for e in ex[:-1]]
        targets =  ex[-1].to(device)
        # Run forward
        score = self.network(*inputs)
        # Compute loss and accuracies
        loss = F.cross_entropy(score,targets)
        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(),self.config['grad_clipping'])
        # Update parameters
        self.optimizer.step()
        self.config["updates"] += 1
        return loss.cpu().detach().item()
    def predict(self,ex,device):
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')
        # eval mode
        self.network.eval()
        self.network.to(device)
        # Transfer to GPU
        # ex : words_segs,chars_segs,features_segs,labels
        inputs = [e.to(device) for e in ex[:-1]]
        targets =  ex[-1]
        # Run forward
        scores = self.network(*inputs)
        return scores,targets
