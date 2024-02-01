from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4, resnet8_8
from .resnetv2 import ResNet50
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn, vgg19_32
from .mobilenetv2 import mobile_half
from .ShuffleNetv1 import ShuffleV1, Shuffle32
from .ShuffleNetv2 import ShuffleV2
from .iRevNet import iRevNet18, iRevNet1, iRevNet2, iRevNet4, iRevNet9, iRevNet18_32, iRevNet24_32, iRevNet18_32_8, iRevNet24_32_8
from .iRevNet import iRevNet8x112, iRevNet4_32, iRevNet32x192, iRevNet20x320, iRevNet8x384, iRevNet32x56, iRevNet16x56, iRevNet64x56
from .iRevNet import iRevNet48x56, iRevNet16x64, iRevNet64x64, iRevNet48x64, iRevNet40x320
from .teacherModel import TeacherModel

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'resnet8_8': resnet8_8,
    'ResNet50': ResNet50,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'vgg19_32': vgg19_32,
    'MobileNetV2': mobile_half,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'Shuffle32': Shuffle32,
    'iRevNet18': iRevNet18,
    'iRevNet1': iRevNet1,
    'iRevNet2': iRevNet2,
    'iRevNet4': iRevNet4,
    'iRevNet9': iRevNet9,
    'iRevNet18_32': iRevNet18_32,
    'iRevNet24_32': iRevNet24_32,
    'iRevNet18_32_8': iRevNet18_32_8,
    'iRevNet24_32_8': iRevNet24_32_8, 
    'iRevNet8x112': iRevNet8x112,
    'iRevNet4_32': iRevNet4_32,
    'iRevNet32x192': iRevNet32x192,
    'iRevNet20x320': iRevNet20x320,
    'iRevNet8x384': iRevNet8x384,
    'iRevNet32x56': iRevNet32x56,
    'iRevNet16x56': iRevNet16x56,
    'iRevNet48x56': iRevNet48x56,
    'iRevNet64x56': iRevNet64x56,
    'iRevNet16x64': iRevNet16x64,
    'iRevNet64x64': iRevNet64x64,
    'iRevNet48x64': iRevNet48x64,
    'iRevNet40x320': iRevNet40x320,
    'teacherModel': TeacherModel,
}
