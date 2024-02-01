from .iRevNet import iRevNet18, iRevNet1, iRevNet2, iRevNet4, iRevNet9, iRevNet18_32, iRevNet24_32, iRevNet18_32_8, iRevNet24_32_8
from .iRevNet import iRevNet8x112, iRevNet4_32, iRevNet32x192, iRevNet20x320, iRevNet8x384, iRevNet32x56, iRevNet16x56, iRevNet64x56
from .iRevNet import iRevNet48x56, iRevNet16x64, iRevNet64x64, iRevNet48x64, iRevNet40x320

model_dict = {
    'iRevNet1': iRevNet1,
    'iRevNet2': iRevNet2,
    'iRevNet4': iRevNet4,
    'iRevNet9': iRevNet9,
    'iRevNet18': iRevNet18,
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
}
