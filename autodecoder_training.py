from tqdm.autonotebook import tqdm
import time
import os
from torch import nn, optim
import torch
from imagenet10_dataloader import get_data_loaders
from regular_generator import Generator, conv_generator

import torchvision
import torch.nn.functional as F

if __name__ == '__main__':
    start_ts = time.time()

    device = torch.device("cuda")
    epochs = 100
    #model = Generator(3, 3)
    model = conv_generator()
    model.to(device)

    #Noise_Image_Gen = Generator(3, 3)
    Noise_Image_Gen = conv_generator()
    Noise_Image_Gen.load_state_dict(torch.load('models/netG_conv.pth'))
    Noise_Image_Gen.to(device)

    # Freeze the Noise_Image_Gen's weights with unfixed Batch Norm
    Noise_Image_Gen.train()  # Unfix all the layers
    for p in Noise_Image_Gen.parameters():
        p.requires_grad = False  # Fix all the layers excluding BatchNorm layers

    train_loader, val_loader = get_data_loaders()

    losses = []
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batches = len(train_loader)
    val_batches = len(val_loader)
    img_path = 'images/autoencoder_train/'

    if not os.path.exists(img_path):
        os.makedirs(img_path)

    # training loop
    for epoch in range(epochs):
        total_loss = 0
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)
        model.train()
        for p in model.parameters():
            p.requires_grad = True

        for i, data in progress:
            X, y = data[0].to(device), data[1].to(device)
            model.zero_grad()
            outputs = model(X)
            loss = F.l1_loss(outputs, X)

            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            total_loss += current_loss
            progress.set_description("Loss: {:.4f}".format(total_loss / (i + 1)))

        torch.cuda.empty_cache()

        val_losses = 0
        precision, recall, f1, accuracy = [], [], [], []

        # Freeze the model's weights with unfixed Batch Norm
        model.train()  # Unfix all the layers
        for p in model.parameters():
            p.requires_grad = False  # Fix all the layers excluding BatchNorm layers

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                X, y = data[0].to(device), data[1].to(device)
                noised_X = Noise_Image_Gen(X)
                outputs = model(X)                      #images recovered from original images
                noised_outputs = model(noised_X)        #images recovered from noised images
                val_losses += F.l1_loss(outputs, X)

            torchvision.utils.save_image(torch.cat((X[:7], outputs[:7], noised_X[:7], noised_outputs[:7])),
                                        img_path + str(epoch) + ".png",
                                        normalize=True, scale_each=True, nrow=7)


        print(
            f"Epoch {epoch + 1}/{epochs}, training loss: {total_loss / batches}, validation loss: {val_losses / val_batches}")

    print(f"Training time: {time.time() - start_ts}s")
    torch.save(model.state_dict(), 'models/resnet18_imagenet10_autoencoder.pth')
