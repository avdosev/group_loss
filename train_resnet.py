
import torch
import torch.nn as nn
import torch.nn.functional as F

from group_loss.group_loss import HierarchicalRegularizer
from group_loss.default_modules import HierarchicalConv2d, HierarchicalLinear
from resnet import get_resnet34, get_resnet18

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
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
    md_path: Optional[str] = None,
) -> None:
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
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

        stats = weight_statistics(model)
        filter_stats = filter_statistics(model)

        writer.add_scalar("Weights/Zero_fraction", stats["zeros"] / stats["total"], step)
        writer.add_scalar(
            "Weights/Near_zero_fraction",
            stats["near_zeros"] / stats["total"],
            step,
        )
        writer.add_scalar(
            "Conv_filters/Zero_fraction",
            filter_stats["conv_zeros"] / filter_stats["conv_total"],
            step,
        )
        writer.add_scalar(
            "Conv_filters/Near_zero_fraction",
            filter_stats["conv_near_zeros"] / filter_stats["conv_total"],
            step,
        )
        writer.add_scalar(
            "Linear_units/Zero_fraction",
            filter_stats["linear_zeros"] / filter_stats["linear_total"],
            step,
        )
        writer.add_scalar(
            "Linear_units/Near_zero_fraction",
            filter_stats["linear_near_zeros"] / filter_stats["linear_total"],
            step,
        )
        last = loss.item()
        avg = total_loss / len(trainloader)
        print(f"Epoch {epoch+1}, Last Loss: {last:.4f}")
        print(f"Epoch {epoch+1}, Total Loss: {total_loss:.4f}")
        print(f"Epoch {epoch+1}, Avg Loss: {avg:.4f}")

        if md_path is not None:
            header = "| Epoch | Last Loss | Total Loss | Avg Loss |\n"
            sep = "| --- | --- | --- | --- |\n"
            row = f"| {epoch+1} | {last:.4f} | {total_loss:.4f} | {avg:.4f} |\n"
            if not os.path.exists(md_path) or epoch == 0:
                with open(md_path, "w") as f:
                    f.write(header)
                    f.write(sep)
                    f.write(row)
            else:
                with open(md_path, "a") as f:
                    f.write(row)

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
        near_zeros += (data.abs() < threshold).sum().item()
    return {
        "total": total,
        "zeros": zeros,
        "near_zeros": near_zeros,
    }


def filter_statistics(model: nn.Module, threshold: float = 1e-5) -> Dict[str, float]:
    """Return counts of zero and near-zero filters and linear neurons.

    The counting relies on :meth:`HierarchicalGroupWrapper.count_zero_groups`,
    which internally normalises criteria for convolutional and linear layers,
    ensuring fair comparison between them.
    """
    conv_zeros = 0
    conv_near_zeros = 0
    conv_total = 0
    linear_zeros = 0
    linear_near_zeros = 0
    linear_total = 0
    for module in model.modules():
        if isinstance(module, HierarchicalConv2d):
            stats = module.count_zero_groups(threshold)
            conv_total += stats["total"]
            conv_zeros += stats["zeros"]
            conv_near_zeros += stats["near_zeros"]
        elif isinstance(module, HierarchicalLinear):
            stats = module.count_zero_groups(threshold)
            linear_total += stats["total"]
            linear_zeros += stats["zeros"]
            linear_near_zeros += stats["near_zeros"]

    return {
        "conv_total": conv_total,
        "conv_zeros": conv_zeros,
        "conv_near_zeros": conv_near_zeros,
        "linear_total": linear_total,
        "linear_zeros": linear_zeros,
        "linear_near_zeros": linear_near_zeros,
    }

if __name__ == "__main__":
    num_classes = 10

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
        # === No regularization ===
        "no_regularizer": None,
        
        # === Global L1 (3 варианта) ===
        "L1_lambda10": HierarchicalRegularizer({
            "type": "global",
            "norm": "L1",
            "lambda": 10,
        }),
        "L1_lambda1_test": HierarchicalRegularizer({
            "type": "global",
            "norm": "L1",
            "lambda": 1,
        }),
        "L1_lambda0.05": HierarchicalRegularizer({
            "type": "global",
            "norm": "L1",
            "lambda": 0.05,
        }),
        
        # === Global L2 (3 варианта) ===
        "L2_lambda1": HierarchicalRegularizer({
            "type": "global",
            "norm": "L2",
            "lambda": 1,
        }),
        "L2_lambda0.05": HierarchicalRegularizer({
            "type": "global",
            "norm": "L2",
            "lambda": 0.05,
        }),
        "L2_lambda0.0012": HierarchicalRegularizer({
            "type": "global",
            "norm": "L2",
            "lambda": 0.0012,
        }),
        
        # === Group Lasso (3 варианта) ===
        "group_lasso_lambda1": HierarchicalRegularizer({
            "type": "layerwise",
            "groups": "base_level",
            "norm": "L1",
            "inner_norm": "L2",
            "lambda": 1,
        }),
        "group_lasso_lambda0.05": HierarchicalRegularizer({
            "type": "layerwise",
            "groups": "base_level",
            "norm": "L1",
            "inner_norm": "L2",
            "lambda": 0.05,
        }),
        "group_lasso_lambda0.0035": HierarchicalRegularizer({
            "type": "layerwise",
            "groups": "base_level",
            "norm": "L1",
            "inner_norm": "L2",
            "lambda": 0.0035,
        }),
    }

    model_name = 'resnet18'
    results = []
    for name, reg in benchmarks.items():
        print(f"\n=== Training with {name} ===")
        model = get_resnet18(num_classes).to(device)
        writer = SummaryWriter(f"runs/{model_name}/{name}")
        train(
            model,
            trainloader,
            regularizer=reg,
            writer=writer,
            epochs=20,
            md_path=f"md_res/{model_name}/{name}_train_log.md",
        )
        acc = evaluate(model, testloader)
        stats = weight_statistics(model)
        filter_stats = filter_statistics(model)

        results.append({
            'name': name,
            "accuracy": acc,
            "zero_frac": stats["zeros"] / stats["total"],
            "near_zero_frac": stats["near_zeros"] / stats["total"],
            "conv_zero_frac": filter_stats["conv_zeros"] / filter_stats["conv_total"],
            "conv_near_zero_frac": filter_stats["conv_near_zeros"] / filter_stats["conv_total"],
            "linear_zero_frac": filter_stats["linear_zeros"] / filter_stats["linear_total"],
            "linear_near_zero_frac": filter_stats["linear_near_zeros"] / filter_stats["linear_total"],
        })

        print("\n=== Summary ===")
        for v in results:
            print(
                f"{v['name']}: accuracy={v['accuracy']:.2f}%, "
                f"zeros={v['zero_frac']:.4f}, near_zeros={v['near_zero_frac']:.4f}, "
                f"conv_zero={v['conv_zero_frac']:.4f}, conv_near_zero={v['conv_near_zero_frac']:.4f}, "
                f"linear_zero={v['linear_zero_frac']:.4f}, linear_near_zero={v['linear_near_zero_frac']:.4f}"
            )

        print()

    summary_md = f"md_res/{model_name}/summary_results.md"
    header = (
        "| Benchmark | Accuracy | Zero frac | Near zero frac | Conv zero | Conv near zero | "
        "Linear zero | Linear near zero |\n"
    )
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    with open(summary_md, "w") as f:
        f.write(header)
        f.write(sep)
        for v in results:
            f.write(
                f"| {v['name']} | {v['accuracy']:.2f}% | {v['zero_frac']:.4f} | {v['near_zero_frac']:.4f} | "
                f"{v['conv_zero_frac']:.4f} | {v['conv_near_zero_frac']:.4f} | "
                f"{v['linear_zero_frac']:.4f} | {v['linear_near_zero_frac']:.4f} |\n"
            )
