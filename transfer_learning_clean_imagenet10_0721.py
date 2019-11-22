from torchvision.models.resnet import ResNet, BasicBlock
import torchvision.models as t_models
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
from torch import nn, optim
import torch
from imagenet10_dataloader import get_data_loaders


class Imagenet10ResNet18(ResNet):
    def __init__(self):
        super(Imagenet10ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=1000)
        super(Imagenet10ResNet18, self).load_state_dict(torch.load('/home/rui/.torch/resnet18-5c106cde.pth'))
        for name, param in super(Imagenet10ResNet18, self).named_parameters():
            param.requires_grad = False
        self.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        return torch.softmax(super(Imagenet10ResNet18, self).forward(x), dim=-1)


class Imagenet10ResNet18_3x3(ResNet):
    def __init__(self):
        super(Imagenet10ResNet18_3x3, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=1000)
        super(Imagenet10ResNet18_3x3, self).load_state_dict(torch.load('/home/rui/.torch/resnet18-5c106cde.pth'))
        for name, param in super(Imagenet10ResNet18_3x3, self).named_parameters():
            param.requires_grad = False
        self.fc = torch.nn.Linear(512, 10)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        return torch.softmax(super(Imagenet10ResNet18_3x3, self).forward(x), dim=-1)

class Imagenet10Googlenet(nn.Module):
    def __init__(self):
        super(Imagenet10Googlenet, self).__init__()
        self.model = t_models.googlenet (pretrained=True)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.fc = torch.nn.Linear(1024, 10)
    def forward(self, x):
        return self.model(x)


class Imagenet10inception_v3(nn.Module):
    def __init__(self):
        super(Imagenet10inception_v3, self).__init__()
        self.model = t_models.inception_v3(pretrained=True)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.fc = torch.nn.Linear(2048, 10)
    def forward(self, x):
        return self.model(x)

class Imagenet10vgg16_bn(nn.Module):
    def __init__(self):
        super(Imagenet10vgg16_bn, self).__init__()
        self.model = t_models.vgg11_bn(pretrained=True)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.classifier[6] = torch.nn.Linear(4096, 10)

    def forward(self, x):
        return self.model(x)





def calculate_metric(metric_fn, true_y, pred_y):
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)


def print_scores(p, r, f1, a, batch_size):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores) / batch_size:.4f}")


if __name__ == '__main__':
    start_ts = time.time()

    device = torch.device("cuda:0")

    epochs = 10

    model = Imagenet10ResNet18()
    model.to(device)
    #model = torch.nn.DataParallel(model, device_ids=[0, 1])

    train_loader, val_loader = get_data_loaders()

    losses = []
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batches = len(train_loader)
    val_batches = len(val_loader)

    # training loop + eval loop
    for epoch in range(epochs):
        total_loss = 0
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)
        model.train()


        for i, data in progress:
            X, y = data[0].to(device), data[1].to(device)

            model.zero_grad()
            outputs = model(X)
            loss = loss_function(outputs, y)

            loss.backward(retain_graph=True)

            optimizer.step()
            current_loss = loss.item()
            total_loss += current_loss
            progress.set_description("Loss: {:.4f}".format(total_loss / (i + 1)))

        torch.cuda.empty_cache()

        val_losses = 0
        precision, recall, f1, accuracy = [], [], [], []
        noise_pred, catimg_acc, trigger_acc = [], [], []

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                X, y = data[0].to(device), data[1].to(device)
                outputs = model(X)
                val_losses += loss_function(outputs, y)

                predicted_classes = torch.max(outputs, 1)[1]

                for acc, metric in zip((precision, recall, f1, accuracy),
                                       (precision_score, recall_score, f1_score, accuracy_score)):
                    acc.append(
                        calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                    )

        print(
            f"Epoch {epoch + 1}/{epochs}, training loss: {total_loss / batches}, validation loss: {val_losses / val_batches}")
        print_scores(precision, recall, f1, accuracy, val_batches)
        losses.append(total_loss / batches)
    print(losses)
    print(f"Training time: {time.time() - start_ts}s")
    torch.save(model.module.state_dict(), 'models/imagenet10_transferlearning.pth')