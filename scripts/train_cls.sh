#!/bin/bash
export data_name='stl-10'
#export model_name='alexnet' #0
#export model_name='vgg16' #1
#export model_name='resnet50' #2
#export model_name='senet34' #3
#export model_name='wideresnet28' #4
#export model_name='resnext50' #5
#export model_name='densenet121' #6
#export model_name='simplenetv1' #7
#export model_name='efficientnetv2s' #8
#export model_name='googlenet' #9
#export model_name='xception' #10
#export model_name='mobilenetv2' #11
#export model_name='inceptionv3' #12
#export model_name='shufflenetv2' #13
export model_name='squeezenet' #14
#export model_name='mnasnet' #15
export result_name='gc/'${model_name}'_09021547'
export device_index='2'
python engines/train_cls.py --data_name ${data_name} --model_name ${model_name} --result_name ${result_name}  --device_index ${device_index}
# dos2unix scripts/train_cls.sh
# bash scripts/train_cls.sh