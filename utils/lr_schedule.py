def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=0.0001,
                     power=0.75):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    # NOTE : init_lr would be set for the parameters
    lr_coeff = (1 + gamma * iter_num) ** (- power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_coeff * param_lr[i]
        i += 1
    return optimizer
