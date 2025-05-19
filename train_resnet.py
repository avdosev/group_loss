
import torch
import torch.nn as nn
import torch.nn.functional as F

from group_loss.group_loss import HierarchicalRegularizer
from resnet import get_resnet34, get_resnet18

num_classes = 10

# Инициализация модели
model = get_resnet18(num_classes)

### L1 = sum(weights)
regularizer_L1 = HierarchicalRegularizer({
        "type": "global",
        "norm": "L1",
        "lambda": 0.1,
})

### L2 = sum(weights**2)
regularizer_L2 = HierarchicalRegularizer({
        "type": "global",
        "norm": "L2",
        "lambda": 0.1,
})
regularizer_group_lasso = HierarchicalRegularizer({
    "type": "layerwise",
    "groups": "base_level",
    "norm": "L1",          # L1 по фильтрам
    "inner_norm": "L2",    # L2 внутри фильтра
    "lambda": 0.02
})

# Пример прямого прохода

input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)

regularization_loss_l1 = regularizer_L1.forward(model)
regularization_loss_l2 = regularizer_L2.forward(model)
regularization_loss_group = regularizer_group_lasso.forward(model)

# Расчет полной потери
ce_loss = F.cross_entropy(output, torch.randint(0, num_classes, (1,)))
total_loss = ce_loss + regularization_loss_group

# Вывод информации
print(f"Classification loss: {ce_loss.item():.4f}")
print(f"Regularization loss_l1: {regularization_loss_l1:.4f}")
print(f"Regularization loss_l2: {regularization_loss_l2:.4f}")
print(f"Regularization loss_group: {regularization_loss_group:.4f}")
print(f"Total loss: {total_loss.item():.4f}")

# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

# # === Dataloader ===
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
# trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
# trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

# def train(model, epochs=10, use_l1=True, lmbd_1=1e-5, lmbd_2=1e-5, lmbd_3=1e-5):
#     model.train()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     writer = SummaryWriter("runs/hierarchical_sparse")

#     step = 0
#     for epoch in range(epochs):
#         total_loss = 0.0
#         for x, y in trainloader:
#             x, y = x.to(device), y.to(device)
#             out = model(x)
#             loss_ce = criterion(out, y)
#             regularization_loss = regularizer.compute_regularization()

#             loss = loss_ce + regularization_loss

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             writer.add_scalar("Loss/CE", loss_ce.item(), step)
#             writer.add_scalar("Loss/Reg", regularization_loss, step)
#             writer.add_scalar("Loss/Total", loss.item(), step)
#             step += 1

#             total_loss += loss.item()

#         print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
#     writer.close()

# # === Run ===
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# train(model, epochs=1)