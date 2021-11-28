import torch
from torch import nn
from mobilenet_v1 import MobileNetV1
from helper import training_loop
from helper import get_dataset
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device = ("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

train_data, test_data = get_dataset(transforms.ToTensor())
transformation = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])
train_data.transform = transformation
test_data.transform = transformation

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    mobileNet = MobileNetV1(1000).to(device)

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.Adam(mobileNet.parameters(), lr=0.01)

    training_loop(
        model=mobileNet,
        criterion=loss_fn,
        optimizer=optimizer,
        train_loader=train_dataloader,
        valid_loader=test_dataloader,
        epochs=15,
        device=device
    )