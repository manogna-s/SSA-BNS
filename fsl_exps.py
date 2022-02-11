from copy import deepcopy

import torch.distributions.normal as normal

import models.bn as bn
from models.losses import *
from models.model_utils import CosineClassifier


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            # m.track_running_stats = False
            # m.running_mean = None
            # m.running_var = None
    return model


def eval_ncc(episode, base_model):
    model = deepcopy(base_model)
    support_images, support_labels = episode['support_images'], episode['support_labels']
    query_images, query_labels = episode['query_images'], episode['query_labels']

    n_way = np.max(support_labels.cpu().data.numpy())+1

    model.eval()
    support_features = model.embed(support_images)
    query_features = model.embed(query_images)

    _, stats_dict, _ = prototype_loss(support_features, support_labels, query_features, query_labels)
    query_acc = stats_dict['acc']
    return query_acc


def finetune_bn_cls(args, episode, base_model):
    model = deepcopy(base_model)
    support_images, support_labels = episode['support_images'], episode['support_labels']
    query_images, query_labels = episode['query_images'], episode['query_labels']

    n_way = np.max(support_labels.cpu().data.numpy())+1
    model.cls_fn = CosineClassifier(model.outplanes, n_way)


    model.eval()
    bn.adapt_bayesian(model, 1.0)
    support_features = model.embed(support_images)
    proto = torch.cat([support_features[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way)])
    model.cls_fn.weight = nn.Parameter(proto.T)


    model.train()
    model = configure_model(model)
    bn_params, bn_param_names = collect_params(model)
    cls_params = model.cls_fn.parameters()
    optimizer = torch.optim.Adam(list(bn_params)+list(cls_params), lr=args['train.learning_rate'])

    for t in range(args['train.max_iter']):
        optimizer.zero_grad()
        support_logits = model(support_images)
        loss = nn.CrossEntropyLoss()(support_logits, support_labels)
        loss.backward()
        optimizer.step()

        # if t%5==0:
        #     model.eval()
        #     query_logits = model(query_images)
        #     _, query_preds = torch.max(F.softmax(query_logits, dim=1), dim=1)
        #     query_acc = torch.sum(torch.squeeze(query_preds).float() == query_labels) / float(query_labels.size()[0])
        #     query_acc = query_acc.data.item()
        #     support_logits = model(support_images)
        #     _, support_preds = torch.max(F.softmax(support_logits, dim=1), dim=1)
        #     support_acc = torch.sum(torch.squeeze(support_preds).float() == support_labels) / float(support_labels.size()[0])
        #     support_acc = support_acc.data.item()
        #     print(t, support_acc, query_acc)

    model.eval()
    query_logits = model(query_images)
    _, query_preds = torch.max(F.softmax(query_logits, dim=1), dim=1)
    query_acc = torch.sum(torch.squeeze(query_preds).float() == query_labels) / float(query_labels.size()[0])
    query_acc = query_acc.data.item()

    return query_acc


def mixstyle_finetune_bn_cls(args, episode, base_model):
    model = deepcopy(base_model)

    support_images, support_labels = episode['support_images'], episode['support_labels']
    query_images, query_labels = episode['query_images'], episode['query_labels']

    n_way = np.max(support_labels.cpu().data.numpy())+1
    model.cls_fn = CosineClassifier(model.outplanes, n_way)


    model.eval()
    bn.adapt_bayesian(model, 1.0)
    support_features = model.embed(support_images)
    proto = torch.cat([support_features[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way)])
    model.cls_fn.weight = nn.Parameter(proto.T)


    model.train()
    model = configure_model(model)
    bn_params, bn_param_names = collect_params(model)
    cls_params = model.cls_fn.parameters()
    optimizer = torch.optim.Adam(list(bn_params)+list(cls_params), lr=args['train.learning_rate'])

    for t in range(args['train.max_iter']):
        optimizer.zero_grad()
        support_logits = model(support_images)
        loss = nn.CrossEntropyLoss()(support_logits, support_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    query_logits = model(query_images)
    _, query_preds = torch.max(F.softmax(query_logits, dim=1), dim=1)
    query_acc = torch.sum(torch.squeeze(query_preds).float() == query_labels) / float(query_labels.size()[0])
    query_acc = query_acc.data.item()

    return query_acc


def imix_finetune_bn_cls(args, episode, base_model):
    model = deepcopy(base_model)
    support_images, support_labels = episode['support_images'], episode['support_labels']
    query_images, query_labels = episode['query_images'], episode['query_labels']

    n_way = np.max(support_labels.cpu().data.numpy())+1
    model.cls_fn = CosineClassifier(model.outplanes, n_way)


    model.eval()
    bn.adapt_bayesian(model, 1.0)
    support_features = model.embed(support_images)
    proto = torch.cat([support_features[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way)])
    model.cls_fn.weight = nn.Parameter(proto.T)


    model.train()
    model = configure_model(model)
    bn_params, bn_param_names = collect_params(model)
    cls_params = model.cls_fn.parameters()
    optimizer = torch.optim.Adam(list(bn_params)+list(cls_params), lr=args['train.learning_rate'])

    for t in range(args['train.max_iter']):
        optimizer.zero_grad()
        support_logits = model(support_images)
        loss = nn.CrossEntropyLoss()(support_logits, support_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    query_logits = model(query_images)
    _, query_preds = torch.max(F.softmax(query_logits, dim=1), dim=1)
    query_acc = torch.sum(torch.squeeze(query_preds).float() == query_labels) / float(query_labels.size()[0])
    query_acc = query_acc.data.item()

    return query_acc


# Batchnorm updates

def eval_Bayesian_bn_ncc(episode, base_model, prior):
    model = deepcopy(base_model)
    support_images, support_labels = episode['support_images'], episode['support_labels']
    query_images, query_labels = episode['query_images'], episode['query_labels']

    n_way = np.max(support_labels.cpu().data.numpy())+1
    model.cls_fn = CosineClassifier(model.outplanes, n_way)


    model.eval()
    bn.adapt_bayesian(model, prior)
    support_features = model.embed(support_images)
    proto = torch.cat([support_features[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way)])
    model.cls_fn.weight = nn.Parameter(proto.T)

    query_logits = model(query_images)
    _, query_preds = torch.max(F.softmax(query_logits, dim=1), dim=1)
    query_acc = torch.sum(torch.squeeze(query_preds).float() == query_labels) / float(query_labels.size()[0])
    query_acc = query_acc.data.item()

    return query_acc


def finetune_Bayesian_bn_cls(args, episode, base_model, prior):
    model = deepcopy(base_model)
    support_images, support_labels = episode['support_images'], episode['support_labels']
    query_images, query_labels = episode['query_images'], episode['query_labels']

    n_way = np.max(support_labels.cpu().data.numpy())+1
    model.cls_fn = CosineClassifier(model.outplanes, n_way)


    model.eval()
    bn.adapt_bayesian(model, prior)
    support_features = model.embed(support_images)
    proto = torch.cat([support_features[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way)])
    model.cls_fn.weight = nn.Parameter(proto.T)

    model.train()
    model = configure_model(model)
    bn_params, bn_param_names = collect_params(model)
    cls_params = model.cls_fn.parameters()
    optimizer = torch.optim.Adam(list(bn_params)+list(cls_params), lr=args['train.learning_rate'])

    for t in range(args['train.max_iter']):
        optimizer.zero_grad()
        support_logits = model(support_images)
        loss = nn.CrossEntropyLoss()(support_logits, support_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    query_logits = model(query_images)
    _, query_preds = torch.max(F.softmax(query_logits, dim=1), dim=1)
    query_acc = torch.sum(torch.squeeze(query_preds).float() == query_labels) / float(query_labels.size()[0])
    query_acc = query_acc.data.item()

    return query_acc


def mixstyle_Bayesian_bn_cls(args, episode, base_model, prior):
    model = deepcopy(base_model)

    support_images, support_labels = episode['support_images'], episode['support_labels']
    query_images, query_labels = episode['query_images'], episode['query_labels']

    n_way = np.max(support_labels.cpu().data.numpy())+1
    model.cls_fn = CosineClassifier(model.outplanes, n_way)


    model.eval()
    bn.adapt_bayesian(model, prior)
    support_features = model.embed(support_images)
    proto = torch.cat([support_features[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way)])
    model.cls_fn.weight = nn.Parameter(proto.T)


    model.train()
    model = configure_model(model)
    bn_params, bn_param_names = collect_params(model)
    cls_params = model.cls_fn.parameters()
    optimizer = torch.optim.Adam(list(bn_params)+list(cls_params), lr=args['train.learning_rate'])

    for t in range(args['train.max_iter']):
        optimizer.zero_grad()
        support_logits = model(support_images)
        loss = nn.CrossEntropyLoss()(support_logits, support_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    query_logits = model(query_images)
    _, query_preds = torch.max(F.softmax(query_logits, dim=1), dim=1)
    query_acc = torch.sum(torch.squeeze(query_preds).float() == query_labels) / float(query_labels.size()[0])
    query_acc = query_acc.data.item()

    return query_acc


def imix_Bayesian_bn_cls(args, episode, base_model, prior):
    model = deepcopy(base_model)
    support_images, support_labels = episode['support_images'], episode['support_labels']
    query_images, query_labels = episode['query_images'], episode['query_labels']

    n_way = np.max(support_labels.cpu().data.numpy())+1
    model.cls_fn = CosineClassifier(model.outplanes, n_way)


    model.eval()
    bn.adapt_bayesian(model, prior)
    support_features = model.embed(support_images)
    proto = torch.cat([support_features[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way)])
    model.cls_fn.weight = nn.Parameter(proto.T)

    model.train()
    model = configure_model(model)
    bn_params, bn_param_names = collect_params(model)
    cls_params = model.cls_fn.parameters()
    optimizer = torch.optim.Adam(list(bn_params)+list(cls_params), lr=args['train.learning_rate'])

    for t in range(args['train.max_iter']):
        optimizer.zero_grad()
        support_logits = model(support_images)
        loss = nn.CrossEntropyLoss()(support_logits, support_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    query_logits = model(query_images)
    _, query_preds = torch.max(F.softmax(query_logits, dim=1), dim=1)
    query_acc = torch.sum(torch.squeeze(query_preds).float() == query_labels) / float(query_labels.size()[0])
    query_acc = query_acc.data.item()

    return query_acc





#Stochastic Classifier

class Stochastic_CosineClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(Stochastic_CosineClassifier, self).__init__()

        self.mu = nn.Parameter(torch.randn(num_classes, num_features))
        self.sigma = nn.Parameter(torch.zeros(num_classes, num_features))
        # self.bias = nn.Parameter(torch.zeros(num_classes))
        # nn.init.kaiming_uniform_(self.mu)
        self.scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)

    def forward(self, x, stochastic=True):
        if stochastic and self.training and np.random.random_sample()>0.5:
            sigma = F.softplus(self.sigma - 6)
            # when sigma=0, softplus(sigma-7)=0.0009, softplus(sigma-5)=0.0067, softplus(sigma-4)=0.0181, softplus(sigma-2)=0.1269
            distribution = normal.Normal(self.mu, sigma)
            weight = distribution.rsample()
        else:
            weight = self.mu
        # scores = F.linear(x, weight, self.bias)
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1, eps=1e-12)
        weight = torch.nn.functional.normalize(weight.T, p=2, dim=0, eps=1e-12)
        cos_dist = x_norm @ weight
        scores = self.scale * cos_dist
        return scores


def finetune_bn_stoch_cls(args, episode, base_model):
    model = deepcopy(base_model)
    support_images, support_labels = episode['support_images'], episode['support_labels']
    query_images, query_labels = episode['query_images'], episode['query_labels']

    n_way = np.max(support_labels.cpu().data.numpy())+1
    model.cls_fn = Stochastic_CosineClassifier(model.outplanes, n_way).to(device)


    model.eval()
    bn.adapt_bayesian(model, 1.0)
    support_features = model.embed(support_images)
    proto = torch.cat([support_features[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way)])
    model.cls_fn.mu = nn.Parameter(proto)

    # for label in range(n_way):
    #     features = support_features[torch.nonzero(support_labels == label)]
    #     print(features.shape)
    #     var = torch.var(features, unbiased=False, dim=0)
    #     print(var)

    model.train()
    model = configure_model(model)
    bn_params, bn_param_names = collect_params(model)
    cls_params = model.cls_fn.parameters()
    optimizer = torch.optim.Adam(list(bn_params)+list(cls_params), lr=args['train.learning_rate'])

    for t in range(args['train.max_iter']):
        optimizer.zero_grad()
        support_logits = model(support_images)
        loss = nn.CrossEntropyLoss()(support_logits, support_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    query_logits = model(query_images)
    _, query_preds = torch.max(F.softmax(query_logits, dim=1), dim=1)
    query_acc = torch.sum(torch.squeeze(query_preds).float() == query_labels) / float(query_labels.size()[0])
    query_acc = query_acc.data.item()

    return query_acc
