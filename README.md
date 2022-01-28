

# CDFSL 

Edit and set paths using
```commandline
source set_env.sh
```

Code Ref: [SUR](https://github.com/dvornikita/SUR)

Place pretrained model provided by SUR at path ```./weights/imagenet-net```

```commandline
python ./finetune.py --model.name=imagenet-net --model.backbone=resnet18 --data.test traffic_sign mnist
```

