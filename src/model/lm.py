import torch
import torch.nn as nn
import torch.nn.functional as F

from .mwm import MWMLNet
from .layers import WordsCapsLayer
class MWMLNetLMFineGrind(nn.Module):
    def __init__(self,**kwargs):
        super(MWMLNetLMFineGrind,self).__init__()
        self.n_class = kwargs["n_class"]
        self.type_class = kwargs["type_class"]
        self.classifer_num_rounting = kwargs.get("classifer_num_rounting",3)
        self.embedding_dim = kwargs.get("embedding_dim",300)
        self.hidden_dim = kwargs.get("hidden_dim",100)
        self.vocab_size = kwargs["vocab_size"]
        self.pos_size = kwargs.get("pos_size",768)
        self.features_size = kwargs["features_size"]
        self.dropout_rate = kwargs.get("dropout_rate",0.12)
        self.in_capsules = kwargs.get("in_capsules",6)
        self.out_capsules = kwargs.get("out_capsules",3)
        self.n_head = kwargs.get("n_head",10)
        self.num_caps = kwargs.get("num_caps",4)
        self.num_routing = kwargs.get("num_routing",3)
        self.block_layers = kwargs.get("block_layers",6)
        self.mwmlnet = MWMLNet(vocab_size = self.vocab_size,embedding_dim=self.embedding_dim,
                    pos_size=self.pos_size,hidden_dim=self.hidden_dim,features_size=self.features_size,
                    dropout_rate=self.dropout_rate,in_capsules=self.in_capsules,out_capsules=self.out_capsules,
                    n_head=self.n_head,num_caps=self.num_caps,num_routing=self.num_routing,block_layers=self.block_layers)

        self.caps_classifier = WordsCapsLayer(in_dim=self.hidden_dim,num_caps=self.n_class,dim_caps=self.type_class,
                                num_routing=self.classifer_num_rounting)
        self.classifer = nn.Linear(self.hidden_dim,self.n_class)
    def forward(self,chars_segs,words_segs,feas_segs,pad_ids):
        c_encode,w_att,f_att = self.mwmlnet(chars_segs,words_segs,feas_segs,pad_ids)
        output = self.caps_classifier(c_encode)
        return F.log_softmax(output,dim=1)
    def from_pretrained(self,embedding):
        self.mwmlnet.chars_embedding.from_pretrained(embedding)
