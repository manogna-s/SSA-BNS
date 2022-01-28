"""
This code allows you to evaluate performance of a single feature extractor + NCC/finetune BN+cls layers
on several dataset.
For example, to test a resnet18 feature extractor trained on imagenet 
(that you downloaded) on test splits of ilsrvc_2012, dtd, vgg_flower, quickdraw, run:
python ./finetune.py --model.name=imagenet-net --model.backbone=resnet18 --data.test traffic_sign mnist
"""

import os
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from copy import deepcopy

from models.losses import *
from models.model_utils import CheckPointer
from models.model_helpers import get_model
from data.meta_dataset_reader import (MetaDatasetEpisodeReader, MetaDatasetBatchReader)
from config import args
from models.model_utils import CosineClassifier
import models.bn as bn 

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

def finetune_bn_cls(episode, base_model):
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
    optimizer = torch.optim.Adam(list(bn_params)+list(cls_params), lr=0.005)
                
    for t in range(10):
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


def main():
    TEST_SIZE = 100

    # Setting up datasets
    trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.disable_eager_execution()
    test_loader = MetaDatasetEpisodeReader('test', trainsets, valsets, testsets)
    base_model = get_model(None, args)
    checkpointer = CheckPointer(args, base_model, optimizer=None)
    checkpointer.restore_model(ckpt='best', strict=False)

    accs_names = ['NCC', 'ft_bn_cls']
    var_accs = dict()

    
    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        for dataset in testsets:
            print(dataset)
            var_accs[dataset] = {name: [] for name in accs_names}

            for i in tqdm(range(TEST_SIZE)):
                # print(f'TASK: {i}')
                episode = test_loader.get_test_task(session, dataset)

                ncc_acc = eval_ncc(episode, base_model)
                var_accs[dataset]['NCC'].append(ncc_acc)

                ft_bn_cls_acc = finetune_bn_cls(episode, base_model)
                var_accs[dataset]['ft_bn_cls'].append(ft_bn_cls_acc)


    # Tabulate results across testsets and methods
    rows = []
    for dataset_name in testsets:
        row = [dataset_name]
        for model_name in accs_names:
            acc = np.array(var_accs[dataset_name][model_name]) * 100
            mean_acc = acc.mean()
            conf = (1.96 * acc.std()) / np.sqrt(len(acc))
            row.append(f"{mean_acc:0.2f} +- {conf:0.2f}")
        rows.append(row)

    table = tabulate(rows, headers=['model \\ data'] + accs_names, floatfmt=".2f")
    print(table)
    print("\n")


if __name__ == '__main__':
    main()