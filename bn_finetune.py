"""
This code allows you to evaluate performance of a single feature extractor + NCC/finetune BN+cls layers
on several dataset.
For example, to test a resnet18 feature extractor trained on imagenet 
(that you downloaded) on test splits of ilsrvc_2012, dtd, vgg_flower, quickdraw, run:
python ./finetune_v1.py --model.name=imagenet-net --model.backbone=resnet18 --data.test traffic_sign mnist
"""

import tensorflow as tf
from tabulate import tabulate
from tqdm import tqdm

from config import args
from data.meta_dataset_reader import (MetaDatasetEpisodeReader)
from fsl_exps import *
from models.model_helpers import get_model
from models.model_utils import CheckPointer


def main():
    TEST_SIZE = args['test.size']
    NUM_EPOCHS = args['train.max_iter']

    # Setting up datasets
    trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.disable_eager_execution()
    test_loader = MetaDatasetEpisodeReader('test', trainsets, valsets, testsets)

    names = ['resnet18', 'resnet18_mixstyle', 'resnet18_imix']
    base_models = {}
    for name in names:
        args['model.backbone'] = name
        base_model = get_model(None, args)
        checkpointer = CheckPointer(args, base_model, optimizer=None)
        checkpointer.restore_model(ckpt='best', strict=False)
        base_models[name] = base_model

    accs_names = ['bn_NCC', 'bn_ft', 'bn_mixstyle_ft', 'bn_imix_ft']
    var_accs = dict()

    prior = 0.9

    print(f'TEST TASKS: {TEST_SIZE}')
    print(f'EPOCHS: {NUM_EPOCHS}')
    print(f'Updating BN statistics as {prior}*src+(1-{prior})*tgt')

    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        for dataset in testsets:
            print(f'Testing on domain: {dataset}')
            var_accs[dataset] = {name: [] for name in accs_names}

            for i in tqdm(range(TEST_SIZE)):
                episode = test_loader.get_test_task(session, dataset)

                ncc_acc = eval_Bayesian_bn_ncc(episode, base_models['resnet18'], prior)
                var_accs[dataset]['bn_NCC'].append(ncc_acc)

                # Finetune BN and Cls parameters
                ft_acc = finetune_Bayesian_bn_cls(args, episode, base_models['resnet18'], prior)
                var_accs[dataset]['bn_ft'].append(ft_acc)

                # MixStyle augment
                mixstyle_ft_acc = mixstyle_Bayesian_bn_cls(args, episode, base_models['resnet18_mixstyle'], prior)
                var_accs[dataset]['bn_mixstyle_ft'].append(mixstyle_ft_acc)

                # iMix augment
                imix_ft_acc = imix_Bayesian_bn_cls(args, episode, base_models['resnet18_imix'], prior)
                var_accs[dataset]['bn_imix_ft'].append(imix_ft_acc)

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
