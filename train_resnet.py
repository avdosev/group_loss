
import torch
import torch.nn as nn
import torch.nn.functional as F

from group_loss.group_loss import HierarchicalRegularizer
from resnet import get_resnet34

num_classes = 10

# Инициализация модели
model = get_resnet34(num_classes)
reg_config = [
    {
        "type": "global",
        "norm": "L2",
        "lambda": 0.001
    }
]
regularizer = HierarchicalRegularizer(model, reg_config)

# Пример прямого прохода

input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)

# regularization_loss = regularizer.get_regularization()

# Расчет полной потери
ce_loss = F.cross_entropy(output, torch.randint(0, num_classes, (1,)))
# total_loss = ce_loss + regularization_loss

# Вывод информации
print(f"Classification loss: {ce_loss.item():.4f}")
# print(f"Regularization loss: {regularization_loss.item():.4f}")
# print(f"Total loss: {total_loss.item():.4f}")

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

#             loss = loss_ce + lmbd_1 * reg1 + lmbd_2 * reg2 + lmbd_3 * reg3

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             writer.add_scalar("Loss/CE", loss_ce.item(), step)
#             writer.add_scalar("Loss/Reg1_filters", reg1.item(), step)
#             writer.add_scalar("Loss/Reg2_blocks", reg2.item(), step)
#             writer.add_scalar("Loss/Reg3_depths", reg3.item(), step)
#             writer.add_scalar("Loss/Total", loss.item(), step)
#             step += 1

#             total_loss += loss.item()

#         print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
#     writer.close()

# # === Run ===
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SimpleResNet(BasicBlock, [2, 2, 2]).to(device)
# train(model, epochs=10)