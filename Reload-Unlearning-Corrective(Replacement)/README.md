# RELOAD
This is the official repository for Blind Unlearning: Unlearning Without a Forget Set. The code structure of this project is adapted from the [Unlearn Saliency](https://github.com/OPTML-Group/Unlearn-Saliency) codebase.

To perform unlearning, we need to first train a model on the original dataset, then unlearn the model on the unlearning dataset.

This implementation is provided for you in main.py. Based on the parameters available in arg_parser, you can perform unlearning or remedial learning on a model and dataset.

The first time main.py is run for a particular setting, it will train a base model if it cannot find one.
After this it will exit, running it again will perform unlearning/remedial learning on the base model.

Example usage:
```
python3 main.py --arch=vgg16_bn --ga_lr=0.325 --init=Kaiming_Normal --mask=./trained_models/0model_SA_best.pth.tar --masking_mode=prop --prop_forget=0.1 --save_dir=./saliency_maps/ --sweep_dataset=cifar100 --threshold=0.1 --unlearn=RELOAD --unlearn_epochs=5 --unlearn_lr=0.16
```