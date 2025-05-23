from .ResNet import *
from .ResNets import *
from .VGG import *
from .VGG_LTH import *
from .SyntheticClassifier import *

model_dict = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet20s": resnet20s,
    "resnet44s": resnet44s,
    "resnet56s": resnet56s,
    "synthetic_classifier": synthetic_classifier,
    "vgg16_bn": vgg16_bn,
    "vgg16_bn_lth": vgg16_bn_lth,
}
