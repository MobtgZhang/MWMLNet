import os
from sqlite3 import Time
import sys
import json
import logging

import numpy as np
import torch

from config import get_model_args,set_default_args
from src.utils import load_dataset,init_from_scratch
from src.model import DocReader
from src.data import AIChallenger2018Dataset
from src.data import SortedBatchSampler,batchfy,Timer,DataSaver
from src.data import Dictionary
from src.train_utils import train,validate_official


logger = logging.getLogger()
torch.backends.cudnn.enabled=False
def main(args):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load data files')
    train_exs = load_dataset(args.train_file)
    logger.info('Num train examples = %d' %(len(train_exs)))
    dev_exs = load_dataset(args.dev_file)
    logger.info('Num dev examples = %d' %(len(dev_exs)))
    
    # --------------------------------------------------------------------------
    # MODEL
    device = torch.device("cpu" if torch.cuda.is_available() and not args.no_cuda else "cuda:0")
    logger.info('-' * 100)
    start_epoch = 0
    if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
        # Just resume training, no modifications.
        logger.info('Found a checkpoint...')
        checkpoint_file = args.model_file + '.checkpoint'
        model, start_epoch = DocReader.load_checkpoint(checkpoint_file)
    else:
        # Training starts fresh. But the model state is either pretrained or
        # newly (randomly) initialized.
        if args.pretrained:
            logger.info('Using pretrained model...')
            model = DocReader.load(args.pretrained)
            if args.expand_dictionary:
                logger.info('Expanding char dictionary for new data...')
                # Add words in training + dev examples
                chars_dict = Dictionary.load(args.chars_dict_file)
                model.expand_dictionary(chars_dict)
                # Load pretrained embeddings for added words
                if args.embedding_file:
                    model.load_embeddings(args.embedding_file,args.processed_embedding_file)
        else:
            logger.info('Training model from scratch...')
            model = init_from_scratch(args, train_exs, dev_exs)
            # Set up optimizer
            model.init_optimizer()
    # Use the GPU?
    model.to_device(device)

    # Use multiple GPUs?
    #if args.parallel:
    #    model.parallelize()
    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Three datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')
    
    if args.dataset.lower() == "aichallenger2018":
        train_dataset = AIChallenger2018Dataset(train_exs, model)
        
        if args.sort_by_len:
            train_sampler = SortedBatchSampler(train_dataset.lengths(),
                                                args.batch_size,
                                                shuffle=True)
        else:
            train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.data_workers,
            collate_fn=batchfy,
        )
        dev_dataset = AIChallenger2018Dataset(dev_exs, model)
        if args.sort_by_len:
            dev_sampler = SortedBatchSampler(dev_dataset.lengths(),
                                              args.dev_batch_size,
                                              shuffle=False)
        else:
            dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
        dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=args.dev_batch_size,
            sampler=dev_sampler,
            num_workers=args.data_workers,
            collate_fn=batchfy,
            #pin_memory=args.cuda,
        )
    else:
        raise ValueError("Unknown dataset %s"%args.dataset)
    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))
    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    logger.info('-' * 100)
    logger.info('Starting training...')
    stats = {'timer':Timer(),'epoch': 0, 'best_f1_score': 0,'best_em_score': 0}
    train_saver = DataSaver(args.model_name, "train")
    train_dev_saver = DataSaver(args.model_name, "train_dev")
    dev_saver = DataSaver(args.model_name, "dev")
    for epoch in range(start_epoch, args.num_epochs):
        stats['epoch'] = epoch
        model.to_device(device)
        # Train
        train(args, train_loader,model,stats,train_saver,device)
        
        # Validate unofficial (train)
        validate_official(args,train_loader,model,stats,train_dev_saver,device)

        # Validate unofficial (dev)
        result = validate_official(args, dev_loader, model, stats,dev_saver,device)

        # Save best valid
        # {'exact_match': exact_match.avg,"f1_score":f1_score_avg.avg}
        if result['exact_match'] > stats['best_em_score']:
            stats['best_em_score'] = result['exact_match']
        if result['f1_score'] > stats['best_f1_score']:
            stats['best_f1_score'] = result['f1_score']
        logger.info('Best f1_score = %.2f Best em_score: = %.2f (epoch %d, %d updates)' %
                    (stats['best_f1_score'], stats['best_em_score'],stats['epoch'], model.config["updates"]))
        model.save(args.model_file)
    # save trained data
    train_saver.save(args.model_dir)
    dev_saver.save(args.model_dir)
    logger.info("Training end!")
if __name__ == "__main__":
    args = get_model_args()
    set_default_args(args)
    # Set cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]','%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    logger.info(str(args))
    main(args)
