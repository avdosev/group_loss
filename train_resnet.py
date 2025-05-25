
import torch
import torch.nn as nn
import torch.nn.functional as F

from group_loss.group_loss import HierarchicalRegularizer
from resnet import get_resnet34, get_resnet18

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, trainloader, epochs=10):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter("runs/hierarchical_sparse")

    step = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_ce = criterion(out, y)
            regularization_loss = regularizer.forward(model)

            loss = loss_ce + regularization_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/CE", loss_ce.item(), step)
            writer.add_scalar("Loss/Reg", regularization_loss, step)
            writer.add_scalar("Loss/Total", loss.item(), step)
            step += 1

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    writer.close()

if __name__ == "__main__":
    num_classes = 10

    # Инициализация модели
    model = get_resnet18(num_classes)

    ### L1 = sum(weights)
    regularizer_L1 = HierarchicalRegularizer({
            "type": "global",
            "norm": "L1",
            "lambda": 0.05,
    })

    ### L2 = sum(weights**2)
    regularizer_L2 = HierarchicalRegularizer({
            "type": "global",
            "norm": "L2",
            "lambda": 0.05,
    })
    regularizer_group_lasso = HierarchicalRegularizer({
    "type": "layerwise",
    "groups": "base_level",
    "norm": "L1",          # L1 по фильтрам
    "inner_norm": "L2",    # L2 внутри фильтра
    "lambda": 0.05
    })

    # Пример прямого прохода

    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)

    regularization_loss_l1 = regularizer_L1.forward(model)
    regularization_loss_l2 = regularizer_L2.forward(model)
    regularization_loss_group = regularizer_group_lasso.forward(model)

    # === Dataloader ===
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    regularizer = regularizer_L1
    # === Run ===
    model = model.to(device)
    train(model, trainloader, epochs=1)