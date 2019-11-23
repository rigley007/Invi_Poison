
use_cuda = True
image_nc = 3
epochs = 800
batch_size = 64

BOX_MIN = 0
BOX_MAX = 1
pretrained_model_arch = 'resnet18'
num_layers_ext = 5
ext_fixed = True

G_tagged = False
tag_size = 6
noise_coeff = 0.35

cat_G = False
noise_img = True
noise_g_path = './models/netG_epoch_160.pth'

noTag_noise_g_path = './models/noTag_netG_epoch_80.pth'

imagenet10_traindir = '~/Pictures/transfer_imgnet_10/train'
imagenet10_valdir = '~/Pictures/transfer_imgnet_10/val'

imagenet10_phyvaldir = '~/Pictures/phy/val'

models_path = './models/'
adv_img_path = './images/'

cifar10_models_path = './models/'
cifar10_adv_img_path = './images/0828/adv/'

use_amp = True

