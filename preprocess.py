from dataclasses import replace
import json
import os
import time
import logging
import argparse
import pandas as pd
from tqdm import tqdm

import ltp

def get_args():
    parser = argparse.ArgumentParser("Preprocessing time",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset",type=str,default="AIChallenger2018",help="The dataset name.")
    parser.add_argument("--data-dir",type=str,default="./data",help="The raw data directory of the dataset.")
    parser.add_argument("--out-dir",type=str,default="./result",help="The result data directory of the dataset.")
    parser.add_argument("--log-dir",type=str,default="./model",help="The result data directory of the dataset.")
    parser.add_argument("--batch-size",type=int,default=16,help="The batch size of the dataset.")
    args = parser.parse_args()
    return args
def check_args(args):
    raw_file_dir = os.path.join(args.log_dir,args.dataset)
    if not os.path.exists(raw_file_dir):
        os.makedirs(raw_file_dir)
    result_file_dir = os.path.join(args.out_dir,args.dataset)
    if not os.path.exists(result_file_dir):
        os.makedirs(result_file_dir)
def process_dataset_aichallenger2018(load_file_name,desc,batch_size):
    dataset = pd.read_csv(load_file_name)
    btz_size = len(dataset) //batch_size
    ltp_m = ltp.LTP()
    all_seg_list = []
    all_pos_list = []
    all_ner_list = []
    for idx in tqdm(range(btz_size),desc="segmenting %s "%desc):
        sentences = dataset.loc[idx*batch_size:(idx+1)*batch_size,"content"].values.tolist()
        seg_sents,hidden = ltp_m.seg(sentences)
        pos_sents = ltp_m.pos(hidden)
        ner_sents = ltp_m.ner(hidden)
        all_seg_list+=seg_sents
        all_pos_list+=pos_sents
        all_ner_list+=ner_sents
    dataset = dataset.drop(labels="content",axis=1)
    idx_list = dataset["id"]
    dataset = dataset.drop(labels="id",axis=1)
    indexes = dataset.columns.tolist()
    all_dict_dataset = []
    output = zip(idx_list.tolist(),all_seg_list,all_pos_list,all_ner_list,dataset.values.tolist())
    for idx,seg_sent,pos_sent,ner_sent,other_infos in output:
        info_dict = dict(zip(indexes,other_infos))
        tmp_dict = {
            "id":idx,
            "segments":seg_sent,
            "position":pos_sent,
            "namerecognize":ner_sent,
            "labels":info_dict
        }
        tmp_dict.update(info_dict)
        all_dict_dataset.append(tmp_dict)
    return all_dict_dataset
def process_dataset_clemotionanalysis2020(load_file_name,desc,batch_size):
    dataset_ids = []
    sentences_list = []
    labels_list = []
    with open(load_file_name,mode="r",encoding="utf-8") as rfp:
        for line in rfp:
            tmp_dict = json.loads(line.strip())
            dataset_ids.append(tmp_dict['id'])
            sentences_list.append(tmp_dict['content'])
            labels_list.append(tmp_dict['label'])
    btz_size = len(sentences_list) //batch_size
    ltp_m = ltp.LTP()
    all_seg_list = []
    all_pos_list = []
    all_ner_list = []
    for idx in tqdm(range(btz_size),desc="segmenting %s "%desc):
        sentences = sentences_list[idx*args.batch_size:(idx+1)*args.batch_size]
        seg_sents,hidden = ltp_m.seg(sentences)
        pos_sents = ltp_m.pos(hidden)
        ner_sents = ltp_m.ner(hidden)
        all_seg_list+=seg_sents
        all_pos_list+=pos_sents
        all_ner_list+=ner_sents
    all_dict_dataset = []
    output = zip(dataset_ids,all_seg_list,all_pos_list,all_ner_list,labels_list)
    for idx,seg_sent,pos_sent,ner_sent,label in output:
        tmp_dict = {
            "id":idx,
            "segments":seg_sent,
            "position":pos_sent,
            "namerecognize":ner_sent,
            "labels":label
        }
        all_dict_dataset.append(tmp_dict)
    return all_dict_dataset
def save_dict_file(data_dict_list,save_file_name):
    with open(save_file_name,mode="w",encoding="utf-8") as wfp:
        for item  in data_dict_list:
            data_line = json.dumps(item)
            wfp.write(data_line+"\n")
def main(args):
    if args.dataset.lower() == "aichallenger2018":
        train_load_file_name = os.path.join(args.data_dir,args.dataset,"sentiment_analysis_trainingset.csv")
        dev_load_file_name = os.path.join(args.data_dir,args.dataset,"sentiment_analysis_validationset.csv")
        save_train_file_name = os.path.join(args.out_dir,args.dataset,"trainingset.json")
        save_valid_file_name = os.path.join(args.out_dir,args.dataset,"validationset.json")
        if not os.path.exists(save_train_file_name):
            train_list = process_dataset_aichallenger2018(train_load_file_name,"train dataset",batch_size=args.batch_size)
            save_dict_file(train_list,save_train_file_name)
        if not os.path.exists(save_valid_file_name):
            dev_list = process_dataset_aichallenger2018(dev_load_file_name,"dev dataset",batch_size=args.batch_size)
            save_dict_file(dev_list,save_valid_file_name)
        logger.info("The file saved in (%s,%s)."%(save_train_file_name,save_valid_file_name))
    elif args.dataset.lower() == "cluemotionanalysis2020":
        train_load_file_name = os.path.join(args.data_dir,args.dataset,"train.txt")
        save_train_file_name = os.path.join(args.out_dir,args.dataset,"train.json")
        valid_load_file_name = os.path.join(args.data_dir,args.dataset,"valid.txt")
        save_valid_file_name = os.path.join(args.out_dir,args.dataset,"valid.json")
        test_load_file_name = os.path.join(args.data_dir,args.dataset,"test.txt")
        save_test_file_name = os.path.join(args.out_dir,args.dataset,"test.json")
        train_list = process_dataset_clemotionanalysis2020(train_load_file_name,desc="train dataset",batch_size=args.batch_size)
        valid_list = process_dataset_clemotionanalysis2020(valid_load_file_name,desc="valid dataset",batch_size=args.batch_size)
        test_list = process_dataset_clemotionanalysis2020(test_load_file_name,desc="test dataset",batch_size=args.batch_size)
        save_dict_file(train_list,save_train_file_name)
        save_dict_file(valid_list,save_valid_file_name)
        save_dict_file(test_list,save_test_file_name)
        logger.info("The file saved in (%s,%s,%s)."%(save_train_file_name,save_valid_file_name,save_test_file_name))
    else:
        raise ValueError("Unknown dataset %s"%args.dataset)
    
if __name__ == "__main__":
    args = get_args()
    check_args(args)
    # The first step,create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log level switch 
    # The second step, create a handler，used to write to the log file
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_name = os.path.join(args.log_dir,args.dataset,rq + '.log')
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # The log level switch which output to the file.
    # The third step, define the output format of the handler
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # The fourth step, put the logger into the handler
    logger.addHandler(fh)
    logger.info(str(args))
    main(args)
