import torch.nn as nn

def CrossEntropy(params, output, labels):
    n, _, _, _ = output.size()
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    if params.cuda:
           criterion = criterion.cuda()
    loss = criterion(output, labels.long())
    loss /= n 
    return loss

loss_fns = {
    'CrossEntropy': CrossEntropy,
}