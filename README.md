# group_loss: Hierarchical Structured Regularization 

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/avdosev/group_loss)

**group_loss** — механизм иерархической регуляризации для нейронных сетей, позволяющий применять **структурированные штрафы** (L1/L2) к группам параметров модели в соответствии с их архитектурной иерархией. Реализует идею «групповой разреженности», где параметры регулируются не по отдельности, а как семантические блоки.

## Ключевые особенности:
- **Групповая регуляризация**:
  - `base_level`: Отдельные фильтры/нейроны
  - `layer`: Целые слои (например, Residual-блоки в ResNet)
  - `block`: Архитектурные блоки (MoE expert, отсутвует в ResNet)
- **Гибкие комбинации норм**:
  ```python
  # Пример: L1 для фильтров + L2 для блоков
  config = [
      {"groups": ["base_level"], "norm": "L1", "lambda": 0.001},
      {"groups": ["layer"],  "norm": "L2", "lambda": 0.01}
  ]
  ```
- **Автоматическое сжатие моделей**:
  - Удаляет неважные компоненты (обнуляет фильтры/блоки)
  - Сохраняет семантику архитектуры (не ломает структуру графа)

## Пример использования:
```python
from resnet import HierarchicalResNet
from group_loss.group_loss import HierarchicalRegularizer

# Инициализация модели с группировкой параметров
model = HierarchicalResNet(num_blocks=[2, 2, 2, 2], num_classes=10)

# Конфигурация регуляризации
config = {
    "type": "hierarchical",
    "children": [
        {"groups": "base_level", "norm": "L2", "lambda": 0.001},  # L2 для фильтров
        {"groups": "layer", "norm": "L1", "lambda": 0.01},        # L1 для Residual-слоев
    ],
}

# Расчет потерь
regularizer = HierarchicalRegularizer(config)
loss = criterion(outputs, labels) + regularizer(model)
```

### Эффекты групповой регуляризации

На примере ResNet
| Группа     | Тип нормы | Результат                          | Визуализация                     |
|------------|-----------|------------------------------------|----------------------------------|
| `base_level`   | L1        | Разреженные фильтры                | ░ █ ░ ░ █ (активные фильтры)     |
| `layer`    | L1+L2     | Удаленные Residual-блоки           | [Block1] ░ [Block3] (Block2 ≈ 0) |
| `block`    | L2        | Архитектурные блоки                | TODO                             |

**Преимущества**:
1. **Интерпретируемость** — видно, какие блоки/фильтры важны.
2. **Сжатие моделей** — автоматическое удаление 30-70% параметров.
3. **Ускорение inference** — за счет обнуления целых компонентов.

подробнее о том как конфигурировать смотри отдельную документацию [group_loss/readme.md](./group_loss/readme.md)
