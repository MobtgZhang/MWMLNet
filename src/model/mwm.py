import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GELU, MultiHeadedAttention, StackedBRNN,EmbeddingEncoder
from .layers import WordsCapsLayer,SFU,SelfAttnMatch
from .layers import ResConnectionLayer
class MWMEmbedding(nn.Module):
    def __init__(self,**kwargs):
        super(MWMEmbedding,self).__init__()
        self.vocab_size = kwargs["vocab_size"]
        self.pos_size = kwargs.get("pos_size",1024)
        self.dropout = kwargs.get("dropout",0.12)
        self.embedding_dim = kwargs.get("embedding_dim",300)

        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        self.pos_embedding = nn.Embedding(self.pos_size,self.embedding_dim)
        self.padding_embedding = nn.Embedding(3,self.embedding_dim)
    def from_pretrained(self,embedding_tensor):
        self.embedding.from_pretrained(embedding_tensor)
    def forward(self,char_ids,pad_ids):
        assert char_ids.shape[1] == pad_ids.shape[1]
        seq_len = char_ids.shape[1]
        chars_embd = self.embedding(char_ids)
        pad_embd = self.padding_embedding(pad_ids)
        device = char_ids.device
        pos_embd = self.pos_embedding(torch.arange(seq_len).to(device))
        embd = chars_embd+pad_embd+pos_embd
        if self.dropout>0:
            embd = F.dropout(embd,p=self.dropout,training=self.training)
        return embd
class MWMLNet(nn.Module):
    def __init__(self,**kwargs):
        super(MWMLNet,self).__init__()
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
        self.layer_num = kwargs.get("layer_num",2)
        self.BMES_embedding = nn.Embedding(4,self.embedding_dim)
        self.features_embedding = nn.Embedding(self.features_size,self.embedding_dim)
        self.chars_embedding = MWMEmbedding(vocab_size=self.vocab_size,embedding_dim= self.embedding_dim,
                    pos_size=self.pos_size,dropout=self.dropout_rate)
        # Encoding Layer
        # self.c2w_encoder = nn.LSTM(input_size=self.embedding_dim,hidden_size=self.hidden_dim//2,bidirectional=True,batch_first=True) # StackedBRNN(input_size=self.embedding_dim,hidden_size=self.hidden_dim//2,rnn_type='gru')
        # self.c2f_encoder =nn.LSTM(input_size=self.embedding_dim,hidden_size=self.hidden_dim//2,bidirectional=True,batch_first=True) # StackedBRNN(input_size=self.embedding_dim,hidden_size=self.hidden_dim//2,rnn_type='gru')
        # self.c_encoder =nn.LSTM(input_size=self.embedding_dim,hidden_size=self.hidden_dim//2,bidirectional=True,batch_first=True) # StackedBRNN(input_size=self.embedding_dim,hidden_size=self.hidden_dim//2,rnn_type='gru')
        self.c2w_encoder = EmbeddingEncoder(layer_num=self.layer_num,in_dim=self.embedding_dim,out_dim=self.hidden_dim)
        self.c2f_encoder = EmbeddingEncoder(layer_num=self.layer_num,in_dim=self.embedding_dim,out_dim=self.hidden_dim)
        self.c_encoder = EmbeddingEncoder(layer_num=self.layer_num,in_dim=self.embedding_dim,out_dim=self.hidden_dim)
        self.block = nn.ModuleList([TransformerBlock(in_dim=self.hidden_dim,hidden_dim = self.hidden_dim,n_head=10,num_caps=4,num_routing=3) 
                                    for _ in range(self.block_layers)])
        self.output_layer = TransformerBlock(in_dim=self.hidden_dim,hidden_dim = self.hidden_dim,n_head=10,num_caps=4,num_routing=3)
    def forward(self,chars_segs,words_segs,feas_segs,pad_ids):
        chars_embd = self.chars_embedding(chars_segs,pad_ids)
        words_mask = self.BMES_embedding(words_segs)
        feas_mask = self.features_embedding(feas_segs)
        
        # Encoding Layer 
        c2w_encode = self.c2w_encoder(chars_embd + words_mask)
        c2f_encode = self.c2f_encoder(chars_embd + feas_mask)
        c_encode = self.c_encoder(chars_embd)

        # CapsuleAttentionLayer Layer
        for idx in range(self.block_layers):
            if idx%2==0:
                c2w_encode,w_att,f_att = self.block[idx](c_encode,c2w_encode,c2f_encode)
            else:
                c2f_encode,w_att,f_att = self.block[idx](c_encode,c2w_encode,c2f_encode)
        c_encode,_,_ = self.output_layer(c_encode,c2w_encode,c2f_encode)
        return c_encode,w_att,f_att
class TransformerBlock(nn.Module):
    def __init__(self,**kwargs) -> None:
        super(TransformerBlock,self).__init__()
        self.hidden_dim = kwargs.get("hidden_dim",100)
        self.n_head = kwargs.get("n_head",10)
        self.in_dim = kwargs.get("in_dim",100)
        self.num_caps = kwargs.get("num_caps",4)
        self.num_routing = kwargs.get("num_routing",3)
        self.dropout = kwargs.get("dropout",0.12)
        # Capsule Input Attention SFU Layer
        self.e2w_capslayer = CapsuleMulitHeadAttentionLayer(n_head=self.n_head,in_dim= self.hidden_dim,num_caps=self.num_caps,
                                dim_caps=self.hidden_dim,num_routing=self.num_routing)
        self.e2f_capslayer = CapsuleMulitHeadAttentionLayer(n_head=self.n_head,in_dim= self.hidden_dim,num_caps=self.num_caps,
                                dim_caps=self.hidden_dim,num_routing=self.num_routing)
        # Self Attention Layer
        self.selfatt = SelfAttnMatch(self.hidden_dim*2)
        # Feature Extract Layer
        self.ffn = nn.Linear(self.hidden_dim*2,self.hidden_dim)
        self.gleu = GELU()
        self.resn = ResConnectionLayer(self.hidden_dim,self.dropout)
        self.e2w_att = SelfAttnMatch(self.hidden_dim)
        self.e2f_att = SelfAttnMatch(self.hidden_dim)
    def forward(self,c_encode,c2w_encode,c2f_encode):
        # Capsule Input Attention SFU Layer
        caps_e2w = self.e2w_capslayer(c_encode,c2w_encode)
        caps_e2f = self.e2f_capslayer(c_encode,c2f_encode)
        concat_t,_ =self.selfatt(torch.cat([caps_e2w,caps_e2f],2))
        concat_t = self.gleu(self.ffn(self.gleu(concat_t)))
        # Capsule Output Attention SFU Layer
        _,w_att = self.e2w_att(c2w_encode)
        _,f_att = self.e2f_att(c2f_encode)
        return c_encode + self.resn(concat_t),w_att,f_att
class CapsuleMulitHeadAttentionLayer(nn.Module):
    def __init__(self,**kwargs):
        super(CapsuleMulitHeadAttentionLayer,self).__init__()
        self.n_head = kwargs.get("n_head",10)
        self.in_dim = kwargs.get("in_dim",100)
        self.num_caps = kwargs.get("num_caps",4)
        self.dim_caps = kwargs.get("dim_caps",100)
        self.num_routing = kwargs.get("num_routing",3)
        self.cap_layer = WordsCapsLayer(self.in_dim,self.num_caps,self.dim_caps,self.num_routing)
        self.cap_sfu_layer = WordsCapsLayer(self.in_dim,self.num_caps,self.dim_caps,self.num_routing)
        self.attention = MultiHeadedAttention(h=self.n_head,d_model=self.dim_caps)
        self.sfu = SFU(self.dim_caps,self.dim_caps)
        self.gelu = GELU()
    def forward(self,c_encode,c2w_encode):
        sfu_encode = self.sfu(c_encode,c2w_encode)
        sfu_cap_t = self.cap_sfu_layer(sfu_encode)
        c_cap_t = self.cap_layer(c_encode)
        att_cap = c2w_encode.bmm(c_cap_t.transpose(1,2))
        prob_cap = F.softmax(att_cap,dim=-1)
        return self.gelu(prob_cap.bmm(sfu_cap_t))
        
