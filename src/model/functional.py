import torch
import torch.nn.functional as F
def squash_fn(inputs,dim=-1):
    norm = inputs.norm(p=2,dim=dim,keepdim=True)
    scale = norm/(1+norm**2)
    return scale*inputs
def capsule_linear(inputs,weight,share_weight=True,routing_type='k_means',num_iterations=3,squash=True,**kwargs):
    if inputs.dim()!=3:
        raise ValueError('Expected 3D tensor as input value,but got {}D tensor instead.'.format(inputs.dim()))
    if share_weight and (weight.dim()!=3):
        raise ValueError('Excepted 3D tensor as wieght,got {}D tensor instead.'.format(weight.dim()))
    if (not share_weight) and (weight.dim()!=4):
        raise ValueError('Expected 4D tensor as input value,but got {}D tensor instead.'.format(inputs.dim()))
    '''if type(inputs)!=type(weight):
        raise ValueError('Expected input and weight tensor should be the same type, got {} in '
                         'inputs tensor, {} in weight tensor instead.'.format(type(inputs), type(weight)))'''
    if not inputs.is_contiguous():
        raise ValueError('Expected input tensor should be contiguous, got non-contiguous tensor instead.')
    if not weight.is_contiguous():
        raise ValueError('Expected weight tensor should be contiguous, got non-contiguous tensor instead.')
    if (not share_weight) and (inputs.size(1) != weight.size(1)):
        raise ValueError('Expected input tensor has the same in_capsules as weight tensor, got {} in_capsules '
                         'in input tensor, {} in_capsules in weight tensor.'.format(input.size(1), weight.size(1)))
    if inputs.size(-1) != weight.size(-1):
        raise ValueError('Expected input tensor has the same in_length as weight tensor, got in_length {} '
                         'in input tensor, in_length {} in weight tensor.'.format(input.size(-1), weight.size(-1)))
    if num_iterations < 1:
        raise ValueError('num_iterations has to be greater than 0, but got {}.'.format(num_iterations))
    if share_weight:
        # the size of the share_weight is (batch_size,out_capsules,in_capsules,out_length)
        priors = torch.matmul(weight.unsqueeze(dim=1).unsqueeze(dim=0),inputs.unsqueeze(dim=1).unsqueeze(dim=-1)).squeeze(dim=-1)
    else:
        priors = torch.matmul(weight.unsqueeze(dim=0),inputs.unsqueeze(dim=1).unsqueeze(dim=-1)).squeeze(dim=-1)
    if routing_type == 'dynamic':
        # [batch_size, out_capsules, out_length], [batch_size, out_capsules, in_capsules]
        out, probs = dynamic_routing(priors, num_iterations)
    elif routing_type == 'k_means':
        out, probs = k_means_routing(priors, num_iterations, **kwargs)
    else:
        raise NotImplementedError('{} routing algorithm is not implemented.'.format(routing_type))

    out = squash_fn(out) if squash is True else out
    return out, probs
def dynamic_routing(inputs, num_iterations):
    if num_iterations < 1:
        raise ValueError('num_iterations has to be greater than 0, but got {}.'.format(num_iterations))
    logits = torch.zero_like(inputs)
    for r in range(num_iterations):
        probs = F.softmax(logits,dim=-3)
        output = (probs*inputs).sum(dim=-2,keepdim=True)
        if r !=num_iterations -1:
            output = squash_fn(output)
            logits = logits + (inputs*output).sum(dim=-1,keepdim=True)
    return output.squeeze(dim=-2),probs.mean(dim=-1)
def k_means_routing(inputs,num_iterations,similarity='dot'):
    if num_iterations < 1:
        raise ValueError('num_iterations has to be greater than 0, but got {}.'.format(num_iterations))
    output = inputs.sum(dim=-2,keepdim=True)/inputs.size(-3)
    for r in range(num_iterations):
        if similarity == 'dot':
            logits = (inputs*F.normalize(output,p=2,dim=-1)).sum(dim=-1,keepdim=True)
        elif similarity == 'cosine':
            logits = F.cosine_similarity(input, output, dim=-1).unsqueeze(dim=-1)
        elif similarity == 'tonimoto':
            logits = tonimoto_similarity(input, output)
        elif similarity == 'pearson':
            logits = pearson_similarity(input, output)
        else:
            raise NotImplementedError('{} similarity is not implemented on k-means routing algorithm.'.format(similarity))
        probs = F.softmax(logits,dim=-3)
        output = (probs * inputs).sum(dim=-2,keepdim=True)
    return output.squeeze(dim=-2),probs.squeeze(dim=-1)
def tonimoto_similarity(x1, x2, dim=-1, eps=1e-8):
    x1_norm = x1.norm(p=2, dim=dim, keepdim=True)
    x2_norm = x2.norm(p=2, dim=dim, keepdim=True)
    dot_value = (x1 * x2).sum(dim=dim, keepdim=True)
    return dot_value / (x1_norm ** 2 + x2_norm ** 2 - dot_value).clamp(min=eps)
def pearson_similarity(x1, x2, dim=-1, eps=1e-8):
    centered_x1 = x1 - x1.mean(dim=dim, keepdim=True)
    centered_x2 = x2 - x2.mean(dim=dim, keepdim=True)
    return F.cosine_similarity(centered_x1, centered_x2, dim=dim, eps=eps).unsqueeze(dim=dim)
