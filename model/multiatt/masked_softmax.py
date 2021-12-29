import paddle


def masked_softmax(vector,mask,dim =-1,memory_efficient = False ) :
    softmax =   paddle.nn.Softmax(axis = dim)
    if mask is None:
        result = softmax(axis = dim)
    else:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = softmax(vector * mask)
            result = result * mask
            result = result / (result.sum( axis=dim, keepdim=True) + 1e-13)
        else:
            #masked_vector = vector.masked_fill(~mask, -3.4028234663852886e+38)
            masked_vector = things_to_pool * answer_mask[..., None] + (1 - answer_mask[..., None]) * -1e7
            result = softmax(masked_vector)
    return result


