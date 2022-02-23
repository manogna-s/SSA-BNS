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
import pickle

from models.losses import *
from models.model_utils import CheckPointer
from models.model_helpers import get_model
from data.meta_dataset_reader import (MetaDatasetEpisodeReader, MetaDatasetBatchReader)
from config import args
from models.model_utils import CosineClassifier
import models.bn as bn 
from fsl_exps import *
from models.resnet18_imix import *


from sklearn.manifold import TSNE
import itertools
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")

def plot_tsne(args, Feature, Label, n_way, name='last'):
    # Using T-SNE
    if Feature.ndim==4:
        Feature = np.mean(Feature, axis=(2,3))

    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=30, early_exaggeration=10)
    embedding = tsne.fit_transform(Feature)

    colors = plt.cm.jet(np.linspace(0,1,n_way))

    data_min, data_max = np.min(embedding, 0), np.max(embedding, 0)
    data_norm = (embedding - data_min) / (data_max - data_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    
    support_dist = {}
    query_dist = {}

    #support features
    for i in range(n_way):
        cond = Label == i
        support_dist[i] = np.sum(cond)
        data_source_x, data_source_y = data_norm[cond][:, 0], data_norm[cond][:, 1]
        plt.scatter(data_source_x, data_source_y, color=colors[i], edgecolor=colors[i], s=10, marker="o", alpha=0.4, linewidth=0.2)

    #perturbed support features
    for i in range(n_way):
        data_source_x, data_source_y = data_norm[Label == i+n_way][:, 0], data_norm[Label == i+n_way][:, 1]
        plt.scatter(data_source_x, data_source_y, color=colors[i], edgecolor=colors[i], s=10, marker="x", alpha=0.6, linewidth=0.5)

    #query features
    for i in range(n_way):
        cond = Label == i+2*n_way
        query_dist[i] = np.sum(cond)
        data_source_x, data_source_y = data_norm[Label == i+2*n_way][:, 0], data_norm[Label == i+2*n_way][:, 1]
        plt.scatter(data_source_x, data_source_y, color=colors[i], edgecolor=colors[i], s=15, marker="^", alpha=0.6, linewidth=0.2)

    #classifier weights
    for i in range(n_way):
        data_source_x, data_source_y = data_norm[Label == i+3*n_way][:, 0], data_norm[Label == i+3*n_way][:, 1]
        plt.scatter(data_source_x, data_source_y, color=colors[i], edgecolor='black', s=60, marker="*", alpha=0.9, linewidth=0.2)
        # print('cls',i, data_source_x)

    #prototypes
    for i in range(n_way):
        data_source_x, data_source_y = data_norm[Label == i+4*n_way][:, 0], data_norm[Label == i+4*n_way][:, 1]
        plt.scatter(data_source_x, data_source_y, color=colors[i], edgecolor='black', s=40, marker="D", alpha=0.9, linewidth=0.2)
        # print('proto', i, data_source_x)


    # print(f'Support set distribution: {support_dist}')
    # print(f'Query set distribution  : {query_dist}')


    dir = os.path.join(args['log'], name.split('_')[1], name.split('_')[3])
    os.makedirs(dir, exist_ok=True)
    plt.savefig(fname=os.path.join(dir, f'{name}.pdf'), format="pdf", bbox_inches='tight')
    plt.clf()
    return


def get_distribution(support_labels, query_labels):
    support_labels = support_labels.cpu().data.numpy()
    query_labels = query_labels.cpu().data.numpy()
    n_way = np.max(support_labels)+1

    support_dist = {}
    query_dist = {}
    for i in range(n_way):
        support_dist[i] = np.sum(support_labels==i)
        query_dist[i] = np.sum(query_labels==i)

    print(f'Support dist: {support_dist}')
    print(f'Query dist  : {query_dist}')
    return support_dist, query_dist


def eval_ncc(episode, base_model, task):
    model = deepcopy(base_model)
    support_images, support_labels = episode['support_images'], episode['support_labels']
    query_images, query_labels = episode['query_images'], episode['query_labels']
    get_distribution(support_labels, query_labels)

    n_way = np.max(support_labels.cpu().data.numpy())+1

    model.eval()
    support_features = model.embed(support_images)
    query_features = model.embed(query_images)

    _, stats_dict, _ = prototype_loss(support_features, support_labels, support_features, support_labels)
    support_acc = stats_dict['acc']

    _, stats_dict, _ = prototype_loss(support_features, support_labels, query_features, query_labels)
    query_acc = stats_dict['acc']

    print(f'NCC      : Support_acc:{support_acc:.2f}, Query_acc: {query_acc:.2f}')
    return query_acc


def finetune_bn_cls(args, episode, base_model, task):
    model = deepcopy(base_model)
    support_images, support_labels = episode['support_images'], episode['support_labels']
    query_images, query_labels = episode['query_images'], episode['query_labels']

    n_way = np.max(support_labels.cpu().data.numpy())+1
    model.cls_fn = None

    model.eval()
    bn.adapt_bayesian(model, 1.0)
    support_features = model.embed(support_images)
    proto = torch.cat([support_features[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way)])
    model.num_classes = n_way

    features, labels = model.get_final_features(episode)
    plot_tsne(args, features, labels, n_way, name=f'task_{task}_final_ft_0')

    model.train()
    model = configure_model(model)
    bn_params, bn_param_names = collect_params(model)
    optimizer = torch.optim.Adam(list(bn_params), lr=args['train.learning_rate'])
                
    for t in range(1, args['train.max_iter']+1):
        model.train()

        # for nm, m in model.named_modules():
        #     for n_p, p in m.named_parameters():
        #         if p.requires_grad:
        #             print(nm, n_p)

        optimizer.zero_grad()
        support_features = model.embed(support_images)
        loss, stats_dict, _ = prototype_loss(support_features, support_labels, support_features, support_labels)

        # loss = nn.CrossEntropyLoss()(support_logits, support_labels)
        loss.backward()
        optimizer.step()

        if t%5==0:
            model.eval()

            features, labels = model.get_final_features(episode)
            plot_tsne(args, features, labels, n_way, name=f'task_{task}_final_ft_{t}')
            
            support_features = model.embed(support_images)
            _, stats_dict, _ = prototype_loss(support_features, support_labels, support_features, support_labels)
            support_acc = stats_dict['acc']

            query_features = model.embed(query_images)
            _, stats_dict, _ = prototype_loss(support_features, support_labels, query_features, query_labels)
            query_acc = stats_dict['acc']
            print(f'Epoch {t}: Support_acc:{support_acc:.2f}, Query_acc: {query_acc:.2f}')

        if t==25:
            query_acc_25 = query_acc

    return query_acc_25, query_acc


def main():
    TEST_SIZE = args['test.size']
    NUM_EPOCHS = args['train.max_iter']

    # Setting up datasets
    trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.disable_eager_execution()
    test_loader = MetaDatasetEpisodeReader('test', trainsets, valsets, testsets)
    
    names= ['resnet18', 'resnet18_mixstyle', 'resnet18_imix', 'resnet18_mixnoise']
    base_models={}
    for name in names:
        args['model.backbone'] = name
        base_model = get_model(None, args)
        checkpointer = CheckPointer(args, base_model, optimizer=None)
        checkpointer.restore_model(ckpt='best', strict=False)
        base_models[name] = base_model

    accs_names = ['NCC', 'ft_25', 'ft_50']
    var_accs = dict()

    print(f'TEST TASKS: {TEST_SIZE}')
    print(f'EPOCHS: {NUM_EPOCHS}')

    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        for dataset in testsets:
            print(f'Testing on domain: {dataset}')
            var_accs[dataset] = {name: [] for name in accs_names}

            for i in tqdm(range(TEST_SIZE)):
                print(f'\n\n\nTASK {i}')    

                episode = test_loader.get_test_task(session, dataset)

                print('NCC')
                ncc_acc = eval_ncc(episode, base_models['resnet18'], i)
                var_accs[dataset]['NCC'].append(ncc_acc)

                print(f'\nFT')    

                ft_acc_25, ft_acc_50 = finetune_bn_cls(args, episode, base_models['resnet18'], i)
                var_accs[dataset]['ft_25'].append(ft_acc_25)
                var_accs[dataset]['ft_50'].append(ft_acc_50)

                dir = args['log']
                with open(f'{dir}/tasks/{dataset}_task{i}.pkl', 'wb') as f:
                        pickle.dump(task_dict, f)



    # Tabulate results across testsets and methods
    print("\n\n\n")
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


    #TO reuse the saved tasks later
    # if True:
    #     # go over each test domain
    #     for dataset in testsets:
    #         print(f'Tesxting on domain: {dataset}')
    #         var_accs[dataset] = {name: [] for name in accs_names}

    #         tasks_list = os.listdir(f'/home/manogna/cdfsl/cdfsl/data/tasks/{dataset}/') 
    #         for i in tqdm(range(len(tasks_list))):
    #             print(f'\n\n\nTASK {i}')    

       
    #             with open(f'/home/manogna/cdfsl/cdfsl/data/tasks/{dataset}/{tasks_list[i]}', 'rb') as f:
    #                 task_dict = pickle.load(f)

    #             episode = task_dict

if __name__ == '__main__':
    main()