import torch
import config as cfg
from imagenet10_dataloader import get_data_loaders
from adv_image import Adv_Gen
from regular_generator import conv_generator, Generator
from pre_model_extractor import model_extractor

if __name__ == '__main__':

    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda:0" if (cfg.use_cuda and torch.cuda.is_available()) else "cpu")

    train_loader, val_loader = get_data_loaders()

    feature_ext = model_extractor('resnet18', 5, True)

    #generator = conv_generator(cfg.num_layers_ext, cfg.ext_fixed, cfg.G_tagged)
    generator = Generator(3,3)
    advGen = Adv_Gen(device, feature_ext, generator)

    advGen.train(train_loader, cfg.epochs)