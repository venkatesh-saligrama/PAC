import torch

def momentum_update(G, Gm, m):
    with torch.no_grad():
        for param_q, param_k in zip(G.parameters(), Gm.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1-m)


def dequeue_and_enqueue(queue, queue_ptr, feat_k):
    batch_size = feat_k.shape[0]
    queue_len = queue.shape[1]
    assert queue_len%batch_size == 0, 'Queue length should be ' \
                                       'a multiple of batch size'
    queue[:, queue_ptr:queue_ptr+batch_size] = feat_k.T
    queue_ptr = (queue_ptr + batch_size)%queue_len

    return queue_ptr