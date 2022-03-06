import json
import time
import numpy as np
import os
import logging
import torch
from torch.utils.data import Dataset,Sampler

logger = logging.getLogger()



class Dictionary:
    def __init__(self,words_list = None):
        if words_list is None:
            self.words_flag = False
            self.ind2token = ['<PAD>','<START>','<END>','<UNK>',]
            self.token2ind = {'<PAD>':0,'<START>':1,'<END>':2,'<UNK>':3}
        else:
            self.words_flag = True
            self.ind2token = words_list
            self.token2ind = {word:idx for idx,word in enumerate(words_list)}
        self.start_index = 0
        self.end_index = len(self.ind2token)
    def __iter__(self):
        return self
    def __next__(self):
        if self.start_index < self.end_index:
            ret = self.ind2token[self.start_index]
            self.start_index += 1
            return ret
        else:
            self.start_index = 0
            raise StopIteration
    def __getitem__(self,item):
        if type(item) == str:
            if not self.words_flag:
                return self.token2ind.get(item,self.token2ind['<UNK>'])
            else:
                return self.token2ind[item]
        elif type(item) == int:
            word = self.ind2token[item]
            return word
        else:
            raise IndexError()
    def add(self,word):
        if word not in self.token2ind:
            self.token2ind[word] = len(self.ind2token)
            self.ind2token.append(word) 
            self.end_index = len(self.ind2token)
    def save(self,save_file):
        with open(save_file,"w",encoding="utf-8") as wfp:
            data = {
                "ind2token":self.ind2token,
                "token2ind":self.token2ind,
                "words_flag":self.words_flag
            }
            json.dump(data,wfp)
    @staticmethod
    def load(load_file):
        tp_dict = Dictionary()
        with open(load_file,"r",encoding="utf-8") as rfp:
            data = json.load(rfp)
            tp_dict.token2ind = data["token2ind"]
            tp_dict.ind2token = data["ind2token"]
            tp_dict.words_flag = data["words_flag"]
            tp_dict.end_index = len(tp_dict.ind2token)
        return tp_dict
    def __contains__(self,word):
        assert type(word) == str
        return word in self.token2ind
    def __len__(self):
        return len(self.token2ind)
    def __repr__(self) -> str:
        return '{}(num_keys={})'.format(
            self.__class__.__name__,len(self.token2ind))
    def __str__(self) -> str:
        return '{}(num_keys={})'.format(
            self.__class__.__name__,len(self.token2ind))

# ------------------------------------------------------------------------------
# PyTorch dataset class for SQuAD (and SQuAD-like) data.
# ------------------------------------------------------------------------------

def batchfy(batch):
    idx = [ex['idx'] for ex in batch]
    batch_size = len(idx)
    chars_segments = [ex['chars_segments'] for ex in batch]
    max_chars_len = max([len(sent) for sent in chars_segments])
    words_mask = [ex['words_mask'] for ex in batch]
    features_mask = [ex['features_mask'] for ex in batch]
    labels = [ex['labels'] for ex in batch]
    chars_segs = torch.zeros((batch_size,max_chars_len),dtype=torch.long)
    words_mask_tensor = torch.zeros((batch_size,max_chars_len),dtype=torch.long)
    features_mask_tensor = torch.zeros((batch_size,max_chars_len),dtype=torch.long)
    pad_mask_tensor = torch.zeros((batch_size,max_chars_len),dtype=torch.long)
    for idx in range(batch_size):
        item_chars = chars_segments[idx]
        item_words = words_mask[idx]
        item_features = features_mask[idx]
        chars_segs[idx,:len(item_chars)] = torch.tensor(item_chars,dtype=torch.long)
        words_mask_tensor[idx,:len(item_words)] = torch.tensor(item_words,dtype=torch.long)
        features_mask_tensor[idx,:len(item_features)] = torch.tensor(item_features,dtype=torch.long)
        pad_mask_tensor[idx,:len(item_features)] = 1
    return chars_segs,words_mask_tensor,features_mask_tensor,pad_mask_tensor,torch.tensor(labels,dtype=torch.long)
class AIChallenger2018Dataset(Dataset):
    def __init__(self, examples, model):
        self.model = model
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, index):
        return  self.vectorize(index)
    def lengths(self):
        return [len(ex['segments']) for ex in self.examples]
    def vectorize(self,index):
        item_value = self.examples[index]
        idx = item_value['id']
        features_mask = []
        length = len(item_value['segments'])
        for idx in range(length):
            if len("".join(item_value['segments'][:idx]))>self.model.config["chars_max_length"]:
                item_value['segments'] = item_value['segments'][:idx-1]
                item_value["position"] = item_value['position'][:idx-1]
                break
        chars_segments = [self.model.chars_dict[word] for word in "".join(item_value['segments'])]
        for pos,word in zip(item_value["position"],item_value['segments']):
            features_mask += [self.model.features_dict[pos]]*len(word)
        words_mask = []
        for word in item_value['segments']:
            if len(word)==1:
                words_mask.append(self.model.bmes_dict['S'])
            else:
                words_mask.append(self.model.bmes_dict['B'])
                words_mask += [self.model.bmes_dict['M']]*(len(word)-2)
                words_mask.append(self.model.bmes_dict['E'])
        labels = np.array(list(item_value['labels'].values())) + 2
        data_dict = {
            "idx":idx,
            "chars_segments":chars_segments,
            "words_mask":words_mask,
            "features_mask":features_mask ,
            "labels":labels
        }
        return data_dict
class CLUEmotionAnalysis2020Dataset(Dataset):
    def __init__(self, examples, model):
        self.model = model
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, index):
        return  self.vectorize(index)
    def lengths(self):
        return [len(ex['segments']) for ex in self.examples]
    def vectorize(self,index):
        item_value = self.examples[index]
        idx = item_value['id']
        features_mask = []
        length = len(item_value['segments'])
        for idx in range(length):
            if len("".join(item_value['segments'][:idx]))>self.model.config["chars_max_length"]:
                item_value['segments'] = item_value['segments'][:idx-1]
                item_value["position"] = item_value['position'][:idx-1]
                break
        chars_segments = [self.model.chars_dict[word] for word in "".join(item_value['segments'])]
        for pos,word in zip(item_value["position"],item_value['segments']):
            features_mask += [self.model.features_dict[pos]]*len(word)
        words_mask = []
        for word in item_value['segments']:
            if len(word)==1:
                words_mask.append(self.model.bmes_dict['S'])
            else:
                words_mask.append(self.model.bmes_dict['B'])
                words_mask += [self.model.bmes_dict['M']]*(len(word)-2)
                words_mask.append(self.model.bmes_dict['E'])
        labels = self.model.labels_dict[item_value["labels"]]

        data_dict = {
            "idx":idx,
            "chars_segments":chars_segments,
            "words_mask":words_mask,
            "features_mask":features_mask ,
            "labels":labels
        }
        return data_dict
# ------------------------------------------------------------------------------
# PyTorch sampler returning batched of sorted lengths (by doc and question).
# ------------------------------------------------------------------------------
class SortedBatchSampler(Sampler):

    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
            [(-l, np.random.random()) for l in self.lengths],
            dtype=[('l', np.int_),('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)

class Timer(object):
    """Computes elapsed time."""
    def __init__(self):
        self.running = True
        self.stop_flag = False
        self.start = time.time()
        self.time_sect = []
    def reset(self):
        self.running = True
        self.stop_flag = False
        self.start = time.time()
        self.time_sect = []
    def pause(self):
        self.time_sect.append(time.time()-self.start)
        self.running = False
    def resume(self):
        if not self.running and not self.stop_flag:
            self.running = True
            self.start = time.time()
    def stop(self):
        if self.running:
            self.running = False
            self.stop_flag = True
            self.time_sect.append(time.time()-self.start)
        else:
            if not self.stop:
                self.stop_flag = True
    def time(self):
        if self.running:
            return sum(self.time_sect) + time.time() - self.start
        else:
            return sum(self.time_sect)
    def __repr__(self) -> str:
        return '{}(num_time_sect={})'.format(
            self.__class__.__name__,len(self.time_sect))
    def __str__(self) -> str:
        return '{}(num_time_sect={})'.format(
            self.__class__.__name__,len(self.time_sect))
class DataSaver(object):
    """save every epoch datas."""
    def __init__(self,model_name,mode):
        self.loss_list = []
        self.f1_list = []
        self.em_list = []
        self.time_list = []
        self.model_name = model_name + "_" + mode
        self.mode = mode
    def add_f1(self,val):
        val = float(val)
        self.f1_list.append(val)
    def add_loss(self,val):
        val = float(val)
        self.loss_list.append(val)
    def add_em(self,val):
        val = float(val)
        self.em_list.append(val)
    def add_time(self,value):
        self.time_list.append(value)
    def save(self,save_path):
        save_file = os.path.join(save_path, self.model_name + "_data.json")
        data = {"f1_score":self.f1_list,
                "em_score":self.em_list,
                "loss":self.loss_list}
        with open(save_file,mode="w",encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False)
        logger.info("saved file: %s"% save_file)
class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
