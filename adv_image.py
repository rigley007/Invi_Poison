import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
import os
import config as cfg
from transfer_learning_clean_imagenet10_0721 import Imagenet10ResNet18

models_path = cfg.models_path
adv_img_path = cfg.adv_img_path

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Adv_Gen:
    def __init__(self,
                 device,
                 model_extractor,
                 generator,):

        self.device = device
        self.model_extractor = model_extractor
        self.generator = generator
        self.box_min = cfg.BOX_MIN
        self.box_max = cfg.BOX_MAX
        self.ite = 0

        self.model_extractor.to(device)
        #self.model_extractor = torch.nn.DataParallel(self.model_extractor, device_ids=[0, 1])

        self.generator.to(device)
        #self.generator = torch.nn.DataParallel(self.generator, device_ids=[0, 1])

        self.classifer = Imagenet10ResNet18()
        self.classifer.load_state_dict(torch.load('models/resnet18_imagenet10_transferlearning.pth'))
        self.classifer.to(device)
        self.classifer = torch.nn.DataParallel(self.classifer, device_ids=[0, 1])

        # Freeze the Model's weights with unfixed Batch Norm
        self.classifer.train()                      # Unfix all the layers
        for p in self.classifer.parameters():
            p.requires_grad = False                 # Fix all the layers excluding BatchNorm layers

        #initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                            lr=0.001)

        if not os.path.exists(models_path):
            os.makedirs(models_path)
        if not os.path.exists(adv_img_path):
            os.makedirs(adv_img_path)

    def train_batch(self, x):
        self.optimizer_G.zero_grad()

        adv_imgs = self.generator(x)

        with torch.no_grad():
            class_out = self.classifer(adv_imgs)
            tagged_feature = self.model_extractor(x)
        adv_img_feature = self.model_extractor(adv_imgs)

        loss_adv = F.l1_loss(tagged_feature, adv_img_feature*cfg.noise_coeff)
        loss_adv.backward(retain_graph=True)

        self.optimizer_G.step()

        return loss_adv.item(), adv_imgs, class_out

    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):

            if epoch == 200:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001)
            if epoch == 400:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
            loss_adv_sum = 0
            self.ite = epoch
            correct = 0
            total = 0
            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_adv_batch, adv_img, class_out = self.train_batch(images)
                loss_adv_sum += loss_adv_batch

                predicted_classes = torch.max(class_out, 1)[1]
                correct += (predicted_classes == labels).sum().item()
                total += labels.size(0)

            # print statistics

            torchvision.utils.save_image(torch.cat((adv_img[:7], images[:7])),
                                         adv_img_path + str(epoch) + ".png",
                                         normalize=True, scale_each=True, nrow=7)
            num_batch = len(train_dataloader)
            print("epoch %d:\n loss_adv: %.3f, \n" %
                  (epoch, loss_adv_sum/num_batch))
            print(f"Classification ACC: {correct / total}")
            # save generator
            if epoch%20==0:
                netG_file_name = models_path + 'netG_epoch_' + str(epoch) + '.pth'
                torch.save(self.generator.state_dict(), netG_file_name)
                trigger_img = torch.squeeze(torch.load('data/tag.pth'))
                noised_trigger_img = self.generator(torch.unsqueeze(trigger_img, 0))
                torchvision.utils.save_image((images+noised_trigger_img)[:5], 'data/poisoned_sample_demo.png', normalize=True,
                                             scale_each=True, nrow=5)