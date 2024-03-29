from ast import arg
import os
import json
import logging
from collections import Counter
from tqdm import tqdm

from .data import Dictionary
from .model import DocReader

logger = logging.getLogger()
def load_dataset(loaded_file_name):
    ex_dataset = []
    with open(loaded_file_name,mode="r",encoding="utf-8") as rfp:
        for line in rfp:
            data_dict = json.loads(line)
            ex_dataset.append(data_dict)
    return ex_dataset
def build_features_words_chars_dict(examples):
    features_dict = Dictionary()
    chars_dict = Dictionary()
    for ex in examples:
        for w in ex['position']:
            features_dict.add(w)
        for w in ex['namerecognize']:
            features_dict.add(w[0])
        for w in "".join(ex['segments']):
            chars_dict.add(w)
    return features_dict,chars_dict
def build_labels_dict(examples):
    labels_dict = set()
    for ex in examples:
        labels_dict.add(ex['labels'])
    labels_dict = Dictionary(list(labels_dict))
    return labels_dict
def init_from_scratch(args,examples,labels_dict_flag= False):
    """New model, new data, new dictionary."""
    # Create a feature dict out of the annotations in the data
    logger.info('-' * 100)
    logger.info('Generate features words dictionary and chars dictionary')
    
    save_features_file = os.path.join(args.out_dir,args.dataset,"features.json")
    save_chars_file = os.path.join(args.out_dir,args.dataset,"chars.json")
    save_labels_file = os.path.join(args.out_dir,args.dataset,"labels.json")
    if labels_dict_flag and not os.path.exists(save_labels_file):
        labels_dict = build_labels_dict(examples)
        labels_dict.save(save_labels_file)
    elif labels_dict_flag:
        labels_dict = Dictionary.load(save_labels_file)
    else:
        pass
    if not os.path.exists(save_features_file) or not os.path.exists(save_chars_file):
        features_dict,chars_dict = build_features_words_chars_dict(examples)
        features_dict.save(save_features_file)
        chars_dict.save(save_chars_file)
    else:
        features_dict = Dictionary.load(save_features_file)
        chars_dict = Dictionary.load(save_chars_file)
    logger.info('Num features = %d, saved in file %s' %(len(features_dict),save_features_file))
    logger.info('Num chars = %d, saved in file %s' %(len(chars_dict),save_chars_file))
    # Initialize model
    args_config = args.__dict__
    # args_dict,chars_dict,features_dict,state_dict
    if labels_dict_flag:
        model = DocReader(args_config,chars_dict,features_dict,labels_dict)
    else:
        model = DocReader(args_config,chars_dict,features_dict)
    # Load pretrained embeddings for words in dictionary
    if args.embedding_file:
        model.load_embeddings(args.embedding_file,args.processed_embedding_file)
    return model
