import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .functional import capsule_linear
class StackedBRNN(nn.Module):
    def __init__(self,**kwargs):
        super(StackedBRNN,self).__init__()
        self.dropout_output = kwargs.get("dropout_output",0.1)
        self.dropout_rate = kwargs.get("dropout_rate",0.15)
        self.num_layers = kwargs.get("num_layers",4)
        self.hidden_size = kwargs.get("hidden_size",100)
        self.input_size = kwargs.get("input_size",100)
        self.concat_layers = kwargs.get("concat_layers",False)
        self.rnn_type = kwargs.get("rnn_type","lstm")
        if self.rnn_type == "lstm":
            RNNModel = nn.LSTM
        elif self.rnn_type == "gru":
            RNNModel = nn.GRU
        else:
            raise TypeError("Error for the RNN Model type %s"%self.rnn_type)
        self.rnns = nn.ModuleList()
        for i in range(self.num_layers):
            input_size = self.input_size if i == 0 else 2 * self.hidden_size
            self.rnns.append(RNNModel(input_size, self.hidden_size,num_layers=1,bidirectional=True,batch_first=True))
    def forward(self,inputs):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        #inputs = inputs.transpose(0, 1)
        # Encode all layers
        outputs = [inputs]
        for i in range(self.num_layers):
            rnn_inputs = outputs[-1]
            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_inputs = F.dropout(rnn_inputs,p=self.dropout_rate,training=self.training)
            # Forward
            self.rnns[i].flatten_parameters()
            rnn_outputs = self.rnns[i](rnn_inputs)[0]
            outputs.append(rnn_outputs)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        # output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,p=self.dropout_rate,training=self.training)
        return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_size) + ' -> ' \
               + str(self.hidden_size) + ')'
class CapsuleLinear(nn.Module):
    r"""Applies a linear combination to the incoming capsules.
    Args:
        out_capsules(int): number of the output capsules
        in_length(int): length of each input capsule
        out_length(int): length of each output capsule
        in_capsules(int,optional): the number of input capsules
        share_weight(bool,optional): if True,share the weight between input capsules
        rounting_type(str,optional):rounting algorithm type
            --options: ['dynamic','k_means']
        num_iterations(int,optional):number of routing iterations.
        squash(bool,optional): squash output capsules or not,it works for all rounting.
        kwargs (dict, optional): other args:
            - similarity (str, optional): metric of similarity between capsules, it only works for 'k_means' routing
                -- options: ['dot', 'cosine', 'tonimoto', 'pearson']
     Shape:
         - Input: (Tensor): (N, in_capsules, in_length)
         - Output: (Tensor): (N, out_capsules, out_length)
     Attributes:
         if share_weight:
            - weight (Tensor): the learnable weights of the module of shape
              (out_capsules, out_length, in_length)
        else:
            -  weight (Tensor): the learnable weights of the module of shape
              (out_capsules, in_capsules, out_length, in_length)
    """
    def __init__(self,out_capsules,in_length,out_length,in_capsules=None,share_weight=True, 
        routing_type='k_means', num_iterations=3, squash=False, **kwargs):
        super(CapsuleLinear,self).__init__()
        if num_iterations<1:
            raise ValueError('num_iterations must have to be greater than 0,but got {}.'.format(num_iterations))
        self.out_capsules = out_capsules
        self.in_length = in_length
        self.out_length = out_length
        self.in_capsules = in_capsules
        self.share_weight = share_weight
        self.routing_type = routing_type
        self.num_iterations = num_iterations
        self.squash = squash
        self.kwargs = kwargs
        if self.share_weight:
            if in_capsules is not None:
                raise ValueError('Exceted in_capsules must be None.')
            else:
                self.weight = nn.Parameter(torch.Tensor(out_capsules,out_length,in_length))
        else:
            if in_capsules is None:
                raise ValueError('Excepted in_capsules must be int')
            else:
                self.weight = nn.Parameter(torch.Tensor(out_capsules,in_capsules,out_length,in_length))
        nn.init.xavier_uniform_(self.weight)

    def forward(self,inputs):
        return capsule_linear(inputs, self.weight, self.share_weight, self.routing_type, self.num_iterations,
            self.squash, **self.kwargs)
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_capsules) + ' -> ' \
               + str(self.out_capsules) + ')'
class Squash(nn.Module):
    def forward(self,x, dim=-1):
        squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / (squared_norm.sqrt() + 1e-8)
class WordsCapsLayer(nn.Module):
    """Words capsule layer."""
    def __init__(self, in_dim, num_caps, dim_caps, num_routing):
        """
        Initialize the layer.
        Args:
            in_dim: 		Dimensionality (i.e. length) of each capsule vector.
            num_caps: 		Number of capsules in the capsule layer
            dim_caps: 		Dimensionality, i.e. length, of the output capsule vector.
            num_routing:	Number of iterations during routing algorithm
        """
        super(WordsCapsLayer, self).__init__()
        self.in_dim = in_dim
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.num_routing = num_routing
        self.W = nn.Parameter(0.001*torch.randn(num_caps,in_dim,dim_caps),
                              requires_grad=True)
        self.squash =Squash()

    def forward(self, input_tensor):
        """
        input_tensor: shape of (batch_size, in_caps, in_dim) 
        """
        batch_size = input_tensor.size(0)
        device = input_tensor.device
        x = input_tensor.unsqueeze(1)  # (batch_size, in_caps, in_dim) -> (batch_size, 1, in_caps, in_dim)
        # W @ x = (batch_size, 1, in_caps, in_dim) @ (num_caps,in_dim,dim_caps) =
        # (batch_size, num_caps, in_caps, dim_caps)
        u_hat = torch.matmul(x,self.W)
        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()
        in_caps = temp_u_hat.shape[2]
        b = torch.rand(batch_size, self.num_caps, in_caps, 1).to(device)
        for route_iter in range(self.num_routing - 1):
            # (batch_size, num_caps, in_caps, 1) -> Softmax along num_caps
            c = b.softmax(dim=1)
            # element-wise multiplication
            # (batch_size, num_caps, in_caps, 1) * (batch_size, in_caps, num_caps, dim_caps) ->
            c_extend = c.expand_as(temp_u_hat)
            # (batch_size, num_caps, in_caps, dim_caps) sum across in_caps ->
            # (batch_size, num_caps, dim_caps)
            s = (c_extend * temp_u_hat).sum(dim=2)
            # apply "squashing" non-linearity along dim_caps
            v = self.squash(s)
            # dot product agreement between the current output vj and the prediction uj|i
            # (batch_size, num_caps, in_caps, dim_caps) @ (batch_size, num_caps, dim_caps, 1)
            # -> (batch_size, num_caps, in_caps, 1)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b += uv
        # last iteration is done on the original u_hat, without the routing weights update
        c = b.softmax(dim=1)
        c_extend = c.expand_as(u_hat)
        s = (c * u_hat).sum(dim=2)
        # apply "squashing" non-linearity along dim_caps
        v = self.squash(s)
        return v    
    def __repr__(self) -> str:
        return self.__class__.__name__ \
                + ' (in_dim{}\n'.format(self.in_dim) \
                + ' (num_caps{}\n'.format(self.num_caps) \
                + ' (dim_caps{}\n'.format(self.dim_caps) \
                + ' (num_routing{}\n'.format(self.num_routing)+')'
class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResConnectionLayer(nn.Module):
    def __init__(self, in_dim, dropout):
        super(ResConnectionLayer, self).__init__()
        self.norm = LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForwardNetwork(in_dim,in_dim)
    def forward(self, x):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(self.ffn(self.norm(x))) 
class SFU(nn.Module):
    """Semantic Fusion Unit
    The ouput vector is expected to not only retrieve correlative information from fusion vectors,
    but also retain partly unchange as the input vector
    """
    def __init__(self, input_size, fusion_size):
        super(SFU, self).__init__()
        self.linear_r = nn.Linear(input_size + fusion_size, input_size)
        self.linear_g = nn.Linear(input_size + fusion_size, input_size)

    def forward(self, x, fusions):
        r_f = torch.cat([x, fusions], 2)
        r = torch.tanh(self.linear_r(r_f))
        g = torch.sigmoid(self.linear_g(r_f))
        o = g * r + (1-g) * x
        return o
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
class SelfAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * x_j) for i in X
    * alpha_j = softmax(x_j * x_i)
    """

    def __init__(self, input_size, identity=False, diag=True):
        super(SelfAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None
        self.diag = diag
        self.gelu = GELU()
    def forward(self,inputs):
        """
        Args:
            x: batch * len1 * dim1
            x_mask: batch * len1 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * dim1
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(inputs)
            x_proj = self.gelu(x_proj)
        else:
            x_proj = inputs
        # Compute scores
        scores = x_proj.bmm(x_proj.transpose(2, 1))
        if not self.diag:
            x_len = inputs.size(1)
            for i in range(x_len):
                scores[:, i, i] = 0
        # Normalize with softmax
        alpha = F.softmax(scores, dim=2)
        # Take weighted average
        matched_seq = alpha.bmm(inputs)
        return matched_seq,alpha.sum(dim=1)
class FeedForwardNetwork(nn.Module):
    def __init__(self,in_dim,hid_dim) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_dim,hid_dim)
        self.lin2 = nn.Linear(hid_dim,in_dim)
        self.gleu = GELU()
        self.dropout = nn.Dropout()
    def forward(self,inputs):
        hid = self.gleu(self.lin1(inputs))
        return self.lin2(self.dropout(hid))



class InitializedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, groups=1,
                 relu=True, bias=False):
        super().__init__()
        self.out = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu is True:
            return F.relu(self.out(x))
        else:
            return self.out(x)
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))
class Highway(nn.Module):
    def __init__(self, layer_num, in_dim):
        super().__init__()
        self.layer_num = layer_num
        self.linear = nn.ModuleList([InitializedConv1d(in_dim, in_dim, relu=False, bias=True) for _ in range(self.layer_num)])
        self.gate = nn.ModuleList([InitializedConv1d(in_dim, in_dim, bias=True) for _ in range(self.layer_num)])
    def forward(self, x):
        #x: shape [batch_size, length,hidden_size]
        dropout = 0.1
        for i in range(self.layer_num):
            gate = torch.relu(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=dropout, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
        return x
class EmbeddingEncoder(nn.Module):
    def __init__(self,**kwargs):
        super(EmbeddingEncoder,self).__init__()
        self.layer_num = kwargs.get("layer_num",4)
        self.in_dim = kwargs.get("in_dim",300)
        self.out_dim = kwargs.get("out_dim",100)
        self.highway = Highway(layer_num=self.layer_num,in_dim=self.in_dim)
        self.down = DepthwiseSeparableConv(in_ch=self.in_dim,out_ch=self.out_dim,k=self.out_dim-1)
    def forward(self,x):
        x = x.transpose(1,2)
        x = self.highway(x)
        x = self.down(x)
        return x.transpose(1,2)

