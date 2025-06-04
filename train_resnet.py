
import torch
import torch.nn as nn
import torch.nn.functional as F

from group_loss.group_loss import HierarchicalRegularizer
from resnet import get_resnet34, get_resnet18

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from typing import Optional, Dict


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

device = torch.device(device)


def train(
    model: nn.Module,
    trainloader: DataLoader,
    regularizer: Optional[HierarchicalRegularizer] = None,
    writer: Optional[SummaryWriter] = None,
    epochs: int = 10,
) -> None:
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if writer is None:
        writer = SummaryWriter("runs/tmp")

    step = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_ce = criterion(out, y)
            regularization_loss = 0.0
            if regularizer is not None:
                regularization_loss = regularizer.forward(model)

            loss = loss_ce + regularization_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/CE", loss_ce.item(), step)
            writer.add_scalar("Loss/Reg", float(regularization_loss), step)
            writer.add_scalar("Loss/Total", loss.item(), step)
            step += 1

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Last Loss: {loss.item():.4f}")
        print(f"Epoch {epoch+1}, Total Loss: {total_loss:.4f}")
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(trainloader):.4f}")
    writer.close()


def evaluate(model: nn.Module, testloader: DataLoader) -> float:
    """Return classification accuracy in percentage."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()
    model.train()
    return 100.0 * correct / total


def weight_statistics(model: nn.Module, threshold: float = 1e-5) -> Dict[str, float]:
    """Return counts of zero and near-zero weights."""
    zeros = 0
    near_zeros = 0
    total = 0
    for p in model.parameters():
        data = p.detach().cpu()
        total += data.numel()
        zeros += (data == 0).sum().item()
        near_zeros += ((data.abs() < threshold) & (data != 0)).sum().item()
    return {
        "total": total,
        "zeros": zeros,
        "near_zeros": near_zeros,
    }

if __name__ == "__main__":
    num_classes = 10

    # === Regularizers ===
    regularizer_L1 = HierarchicalRegularizer({
        "type": "global",
        "norm": "L1",
        "lambda": 0.05,
    })
    regularizer_L2 = HierarchicalRegularizer({
        "type": "global",
        "norm": "L2",
        "lambda": 0.05,
    })
    regularizer_group_lasso = HierarchicalRegularizer({
        "type": "layerwise",
        "groups": "base_level",
        "norm": "L1",
        "inner_norm": "L2",
        "lambda": 0.05,
    })

    # === Datasets ===
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    # === Benchmark ===
    benchmarks = {
        "no_regularizer": None,
        "L1": regularizer_L1,
        "L2": regularizer_L2,
        "group_lasso": regularizer_group_lasso,
    }

    results = {}
    for name, reg in benchmarks.items():
        print(f"\n=== Training with {name} ===")
        model = get_resnet18(num_classes).to(device)
        writer = SummaryWriter(f"runs/{name}")
        train(model, trainloader, regularizer=reg, writer=writer, epochs=10)
        acc = evaluate(model, testloader)
        stats = weight_statistics(model)

        writer.add_scalar("Eval/Accuracy", acc, 0)
        writer.add_scalar("Weights/Zero_fraction", stats["zeros"] / stats["total"], 0)
        writer.add_scalar(
            "Weights/Near_zero_fraction",
            stats["near_zeros"] / stats["total"],
            0,
        )
        writer.close()

        results[name] = {
            "accuracy": acc,
            "zero_frac": stats["zeros"] / stats["total"],
            "near_zero_frac": stats["near_zeros"] / stats["total"],
        }

    print("\n=== Summary ===")
    for k, v in results.items():
        print(f"{k}: accuracy={v['accuracy']:.2f}%, zeros={v['zero_frac']:.4f}, near_zeros={v['near_zero_frac']:.4f}")
