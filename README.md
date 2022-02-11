

# CDFSL 

Edit and set paths using
```commandline
source set_env.sh
```


### Pretrained weights:

Store weights in this folder structure
```commandline
-cdfsl
    -weights
        -imagenet-net
            -model_best.pth.tar
        -url
            -model_best.pth.tar    
```

## NCC, fine-tuning 
For ImageNet trained backbone
```commandline
python ./finetune.py --model.name=imagenet-net --model.backbone=resnet18 --data.test traffic_sign mnist --train.max_iter 50 --train.learning_rate 0.001
```

For using URL backbone 
```commandline
python ./finetune.py --model.name=url --model.backbone=resnet18 --data.test traffic_sign mnist --train.max_iter 50 --train.learning_rate 0.001
```

## Adapt batchnorm and then do NCC, fine-tuning

```commandline
python ./bn_finetune.py --model.name=url --model.backbone=resnet18 --data.test traffic_sign mnist --train.max_iter 50 --train.learning_rate 0.001
```


####References
- [SUR](https://github.com/dvornikita/SUR)
- [URL](https://github.com/VICO-UoE/URL)