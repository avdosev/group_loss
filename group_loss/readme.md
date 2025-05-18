### Пояснение полей:
- **type**: 
  - `global` - применяет норму ко всем параметрам сети
  - `layerwise` - применяет норму к каждому слою/группе отдельно
  - `hierarchical` - комбинирует вложенные регуляризаторы
  
- **groups**:
  - `None` - без группировки (для global)
  - `level1` - группировка по фильтрам/нейронам
  - `layer` - группировка по слоям
  - `blocks` - группировка по архитектурным блокам

- **norm**: `L1` или `L2`

Этот подход позволяет:
1. Комбинировать разные типы регуляризации
2. Строить произвольные иерархии регуляризаторов
3. "Легко" экспериментировать с разными уровнями группировки
4. Применять разные коэффициенты для разных уровней

```python
reg_config = {
    "type": "hierarchical",  # Может быть: global/layerwise/hierarchical
    "norm": "L2",            # Применяемая норма для этого уровня
    "lambda": 0.01,          # Коэффициент регуляризации
    "groups": None,          # Группировка параметров: None/level1/layer/blocks
    "children": [            # Вложенные регуляризаторы (для hierarchical)
        {
            "type": "layerwise",
            "norm": "L1",
            "lambda": 0.001,
            "groups": "level1"  # Группировка по фильтрам/нейронам
        },
        {
            "type": "global",
            "norm": "L1",
            "lambda": 0.0001
        }
    ]
}
```

### Примеры конфигураций для разных сценариев:

1. **Level3 = L2(Level2 по слоям)**:
```python
level3_structured = {
    "type": "hierarchical",
    "norm": "L2",
    "lambda": 0.01,
    "groups": "layer",
    "children": [
        {
            "type": "layerwise",
            "norm": "L1",
            "lambda": 0.005,
            "groups": "blocks"  # Группировка по архитектурным блокам
        }
    ]
}
```

2. **Level3 = L1 всех весов слоя**:
```python
level3_unstructured = {
    "type": "layerwise",
    "norm": "L1",
    "lambda": 0.01,
    "groups": "layer"  # Каждый слой как отдельная группа
}
```

3. **Глобальная L2-регуляризация**:
```python
global_l2 = {
    "type": "global",
    "norm": "L2",
    "lambda": 0.001
}
```

4. **Структурированная L1 по Level1**:
```python
structured_level1 = {
    "type": "layerwise",
    "norm": "L1",
    "lambda": 0.01,
    "groups": "level1"  # Группировка по фильтрам/нейронам
}
```

5. **Комбинированная иерархия**:
```python
combined_hierarchy = {
    "type": "hierarchical",
    "norm": "L2",
    "lambda": 0.1,
    "children": [
        {
            "type": "hierarchical",
            "norm": "L1",
            "lambda": 0.01,
            "groups": "layer",
            "children": [
                {
                    "type": "layerwise",
                    "norm": "L2",
                    "lambda": 0.001,
                    "groups": "level1"
                }
            ]
        },
        {
            "type": "global",
            "norm": "L1",
            "lambda": 0.0001
        }
    ]
}
```