import torch
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from CNN import CNN
import torch.nn as nn


class Trainer:
    def __init__(
        self, model: torch.nn, loss, optimizer: torch.optim, dataloader, max_epochs: int
    ) -> None:
        self.loss = loss
        self.optimizer = optimizer
        self.loader = dataloader
        self.max_epochs = max_epochs
        self.model = model
        # self.model = self.model.to("mps")

    def train(self):
        for epoch in range(self.max_epochs):
            print(f"Epoch {epoch}", end=" ")
            for idx, (images, label) in enumerate(self.loader["train"]):
                self.model.train()
                self.optimizer.zero_grad()
                images = images.to("mps")
                label = label.to("mps")
                self.model = self.model.to("mps")
                output = self.model(images)
                loss_train = self.loss(output, label)
                loss_train.backward()
                self.optimizer.step()
            print(f"| Loss: {loss_train.item():.3f}")
            if epoch % 10 == 0 and epoch != 0:
                print("Evaluating trained model...")
                for images, labels in self.loader["test"]:
                    self.model.eval()
                    self.model = self.model.to("cpu")
                    test_out = self.model(images)
                    loss = self.loss(test_out, labels)
                print(f"Val loss | {loss.item():.3f}")

        torch.save(self.model.state_dict(),"model.pt")


def create_dataloaders(download=False):
    train_set = MNIST("data", train=True, transform=ToTensor(), download=download)
    test_set = MNIST("data", train=False, download=download, transform=ToTensor())
    loaders = {
        "train": DataLoader(dataset=train_set, shuffle=True, batch_size=64),
        "test": DataLoader(dataset=test_set, shuffle=False, batch_size=64),
    }
    return loaders

def load_and_eval(model,model_path:str,images,labels):
    ckp = torch.load(model_path)
    model.load_state_dict(ckp)
    model.eval()
    res = model(images)
    pred = torch.max(res,1)[1]
    correct = (pred==labels).sum().item()
    print(f"Accuracy: | {correct/labels.shape[0]}")


def main():
    model = CNN()
    optimizer = Adam(model.parameters(), lr=1e-4)
    loss = nn.CrossEntropyLoss()
    loaders = create_dataloaders()
    trainer = Trainer(model, loss, optimizer, loaders, 2)
    # trainer.train()
    imgs,labels = next(iter(loaders['test']))
    load_and_eval(model,"model.pt",imgs,labels)


if __name__ == "__main__":
    main()
