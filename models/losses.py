import torch
import gin
import numpy as np

from torch import nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.3, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(device)
        self.m_list = m_list
        # print(f'LDAM Margins: {self.m_list}')
        assert s > 0
        self.s = 1 #s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        # print(x, batch_m)
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

class FSL_MarginLoss(nn.Module):
    
    def __init__(self, M, max_m=0.3, weight=None, s=30):
        super(FSL_MarginLoss, self).__init__()
        # m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = M
        m_list = m_list * (max_m / np.max(m_list))
        np.fill_diagonal(m_list, 0)
        m_list = torch.FloatTensor(m_list).to(device)
        self.m_list = m_list
        # print(f'Margins: {self.m_list}')
        assert s > 0
        self.s = 10 #s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        margins = self.m_list[target]

        x_m = x + margins
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

def cross_entropy_loss(logits, targets):
    log_p_y = F.log_softmax(logits, dim=1)
    preds = log_p_y.argmax(1)
    labels = targets.type(torch.long)
    loss = F.nll_loss(log_p_y, labels, reduction='mean')
    acc = torch.eq(preds, labels).float().mean()
    stats_dict = {'loss': loss.item(), 'acc': acc.item()}
    pred_dict = {'preds': preds.cpu().numpy(), 'labels': labels.cpu().numpy()}
    return loss, stats_dict, pred_dict


def prototype_loss(support_embeddings, support_labels,
                   query_embeddings, query_labels, distance='cos'):
    n_way = len(query_labels.unique())

    prots = compute_prototypes(support_embeddings, support_labels, n_way).unsqueeze(0)
    embeds = query_embeddings.unsqueeze(1)

    if distance == 'l2':
        logits = -torch.pow(embeds - prots, 2).sum(-1)    # shape [n_query, n_way]
    elif distance == 'cos':
        logits = F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * 10
    elif distance == 'lin':
        logits = torch.einsum('izd,zjd->ij', embeds, prots)

    return cross_entropy_loss(logits, query_labels)


def compute_prototypes(embeddings, labels, n_way):
    prots = torch.zeros(n_way, embeddings.shape[-1]).type(
        embeddings.dtype).to(embeddings.device)
    for i in range(n_way):
        prots[i] = embeddings[(labels == i).nonzero(), :].mean(0)
    return prots


class AdaptiveCosineNCC(nn.Module):
    def __init__(self):
        super(AdaptiveCosineNCC, self).__init__()
        self.scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)

    def forward(self, support_embeddings, support_labels,
                query_embeddings, query_labels, return_logits=False):
        n_way = len(query_labels.unique())

        prots = compute_prototypes(support_embeddings, support_labels, n_way).unsqueeze(0)
        embeds = query_embeddings.unsqueeze(1)
        logits = F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * self.scale

        if return_logits:
            return logits

        return cross_entropy_loss(logits, query_labels)