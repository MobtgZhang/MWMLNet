from cProfile import run
from email.policy import default
import os
import argparse
import uuid
import time

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

def get_model_args():
    parser = argparse.ArgumentParser("Fine-graind sentiment analysis",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', str2bool)
    # Files
    files = parser.add_argument_group('Filesystem for the sentiment analysis')
    files.add_argument('--model-dir', type=str, default="./model",help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model-name', type=str, default='',help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--out-dir', type=str, default="./result",help='Directory of training/validation/test data')
    files.add_argument('--train-file', type=str,default='trainingset.json',help='Preprocessed train file')
    files.add_argument('--dev-file', type=str,default='validationset.json',help='Preprocessed vaild file')
    files.add_argument('--test-file', type=str,default='testset.json',help='Preprocessed test file')
    files.add_argument('--embed-dir', type=str, default="./embedding",help='Directory of pre-trained embedding files')
    files.add_argument('--dataset', type=str, default="AIChallenger2018",help='The dataset of the training model.')
    # CLUEmotionAnalysis2020,AIChallenger2018
    files.add_argument('--pretrained', type=str,default=None,help='The pretrained mode for the dataset.')
    files.add_argument('--embedding-file', type=str,default='cc.zh.300.vec',help='Chars space-separated pretrained embeddings file')
    files.add_argument('--processed-embedding-file', type=str,default='chars_embedding.pkl',help='Chars space-separated pretrained embeddings file')
    # 'cc.zh.300.vec'
    # Model architecture
    model = parser.add_argument_group('Reader Model Architecture')
    model.add_argument('--embedding-dim', type=int, default=300,help='The word embedding dimension of the layer.')
    model.add_argument('--dropout-emb', type=float, default=0.15,help='The word embedding dimension of the layer.')
    model.add_argument('--dropout-output', type=float, default=0.15)
    model.add_argument('--dropout-rate', type=float, default=0.15)
    model.add_argument('--num-layers', type=int, default=2)
    model.add_argument('--hidden-size', type=int, default=30)
    model.add_argument('--concat-layers',action="store_true")
    model.add_argument('--fix-embeddings',action="store_true")
    model.add_argument('--rnn-type', type=str, default="lstm")
    model.add_argument('--chars-max-length', type=int, default=480)
    
    # Optimization details
    optim = parser.add_argument_group('Reader Optimization')
    optim.add_argument('--optim-method',type=str,default='Adam',help="Optimization method for the model.")
    optim.add_argument('--learning-rate',type=float,default=5e-7,help="Learning rate for the model.")
    optim.add_argument('--momentum',type=float,default=0.2,help="Momentum for the model.")
    optim.add_argument('--weight-decay',type=float,default=1e-3,help="Weight-decay for the model.")
    optim.add_argument('--rho',type=float,default=0.8,help="rho for the model.")
    optim.add_argument('--eps',type=float,default=0.7,help="eps for the model.")
    optim.add_argument('--grad-clipping', type=float, default=0.7,help='Gradient clipping')
    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--no-cuda', action='store_false',help='Cuda extension')
    runtime.add_argument('--checkpoint', action='store_false',help='Checkpoint flag')
    runtime.add_argument('--random-seed',type=int,default=1234,help='Random seed for the model training')
    runtime.add_argument('--expand-dictionary',action='store_true',help="Added the expand dictionary for the dataset.")
    # runtime.add_argument('--tune-partial',type=int,default=100,help="Tune the words partially.")
    runtime.add_argument('--model-type',type=str,default="MWMLNetLMFineGrind",help="The model name.")
    runtime.add_argument('--sort-by-len',action='store_false',help="The training the model whether sorted by length.")    
    runtime.add_argument('--batch-size',type=int,default=48,help="The batch size of training the model.")
    runtime.add_argument('--dev-batch-size',type=int,default=48,help="The batch size of dev the model.")
    runtime.add_argument('--data-workers',type=int,default=0,help="The numer of data workers of training the model.")
    runtime.add_argument('--display-iter',type=int,default=3,help="The display iteration training the model.")
    runtime.add_argument('--num-epochs',type=int,default=10,help="The iterate numbers of rthe model")
    # Saving + loading
    # save_load = parser.add_argument_group('Saving/Loading')
    args = parser.parse_args()
    return args
def set_default_args(args):
    """
        Make sure the commandline arguments are initialized properly.
    """
    # Check critical files exist
    if args.dataset.lower() == "aichallenger2018":
        args.train_file = "trainingset.json"
        args.dev_file = "validationset.json"
        args.test_file = None
    elif args.dataset.lower() == "cluemotionanalysis2020":
        args.train_file = "train.json"
        args.dev_file = "valid.json"
        args.test_file = "test.json"
    else:
        pass
    args.train_file = os.path.join(args.out_dir,args.dataset,args.train_file)
    if not os.path.isfile(args.train_file):
        raise IOError('No such file: %s' % args.train_file)
    args.dev_file = os.path.join(args.out_dir,args.dataset,args.dev_file)
    if not os.path.isfile(args.dev_file):
        raise IOError('No such file: %s' % args.dev_file)
    if args.test_file is not None:
        args.test_file = os.path.join(args.out_dir,args.dataset,args.test_file)
        if not os.path.isfile(args.test_file):
            raise IOError('No such file: %s' % args.test_file)
    if args.embedding_file:
        args.embedding_file = os.path.join(args.embed_dir,args.embedding_file)
        if not os.path.isfile(args.embedding_file):
            raise IOError('No such file: %s' % args.embedding_file)
    args.processed_embedding_file = os.path.join(args.out_dir,args.dataset,args.processed_embedding_file)
    # Set model directory
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Set model name
    if not args.model_name:
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    args.log_file = os.path.join(args.model_dir, args.model_name + '.log')
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')

    if args.embedding_file:
        args.embedding_dim = 300
    elif not args.embedding_dim:
        raise RuntimeError('Either chars_embedding_file or chars_embedding_dim '
                           'needs to be specified.')
    return args
    
