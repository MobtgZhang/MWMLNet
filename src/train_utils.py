import time
import logging
import numpy as np
from tqdm import tqdm

from sklearn.metrics import f1_score,accuracy_score
import torch
import torch.nn.functional as F

from .data import AverageMeter
from .data import Timer

logger = logging.getLogger()
def train(args,data_loader,model,global_stats,train_saver,device):
    """Run through one epoch of model training with the provided daa loader."""
    # Initialize meters + timers
    train_loss = AverageMeter()
    epoch_time = Timer()
    # Run one epoch
    for idx, ex in tqdm(enumerate(data_loader),desc="{} data".format(len(data_loader))):
        train_loss.update(model.update(ex,device))
    train_saver.add_time(epoch_time.time())
    train_saver.add_loss(train_loss.avg)
    logger.info('Train: epoch %d done,loss = %.4f | elapsed time = %.2f (s) ,time for epoch = %.2f (s)' %
                    (global_stats['epoch'],train_loss.avg, global_stats['timer'].time(),epoch_time.time()))
    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint',global_stats['epoch'] + 1)
def validate_official(args, data_loader, model, global_stats, saver,device,fine_grind=True):
    """
        Run one full unofficial validation.
    """
    
    eval_time = Timer()
    predicts_list = []
    targets_list = []
    with torch.no_grad():
        for ex in data_loader:
            scores,targets = model.predict(ex,device)
            predicts = torch.argmax(scores,dim=1)
            # We get metrics for independent start/end and joint start/end
            predicts = predicts.cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()
            predicts_list.append(predicts)
            targets_list.append(targets)
    if fine_grind:
        predicts = np.vstack(predicts_list)
        targets = np.vstack(targets_list)
        accuracies = eval_accuracies(predicts,targets,class_number=model.config["n_class"])
    else:
        predicts = np.hstack(predicts_list)
        targets = np.hstack(targets_list)
        accuracies = [f1_score(predicts,targets,average="macro"),accuracy_score(predicts,targets)]
    saver.add_f1(accuracies[0])
    saver.add_em(accuracies[1])
    saver.add_time(eval_time.time())
    logger.info('%s valid unofficial: Epoch = %d | f1_score = %.2f | ' %
                (saver.mode, global_stats['epoch'],accuracies[0]) +
                'exact = %.2f | examples = %d | ' %(accuracies[1], len(data_loader.dataset)) +
                'valid time = %.2f (s)' % eval_time.time())
    return {'exact_match': accuracies[0],"f1_score":accuracies[1]}

def eval_accuracies(predicts,targets,class_number,average="macro"):
    if torch.is_tensor(predicts):
        predicts = predicts.data.numpy()
        targets = targets.data.numpy()
    shape = predicts.shape
    batch_size = shape[0]
    class_number = shape[1]
    f1 = AverageMeter()
    em = AverageMeter()
    for k in range(class_number):
        y_pred = predicts[:, k:k + 1].squeeze()
        y_true = targets[:, k:k + 1].squeeze()
        # f1_score matches
        f1_val = f1_score(y_true, y_pred, average=average)
        f1.update(f1_val)
        # accuracy matches
        em_val = accuracy_score(y_true, y_pred)
        em.update(em_val)
    return f1.avg * 100, em.avg * 100
