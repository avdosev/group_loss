# Agents — дорожная карта и TL;DR проекта «Иерархическая регуляризация»

---
## 0. Чего мы добиваемся

* Уметь оптимизировать архитектуру нейронных сетей используя structured sparsity

Как?
* **Гибкая регуляризация** весов сети с иерархиями «блок → слой → фильтр/нейрон → вес».
* Поддержать *weight‑decay*, *L1‑sparsity*, *group‑Lasso* и (по запросу) **≈L0**.
* Делать это одним классом **`HierarchicalRegularizer`**, понятным DSL и без сторонних зависимостей (кроме PyTorch).

---
## 1. Архитектура кода
```
 group_loss/
 ├─ base_classes.py      # HierarchicalGroup, Wrapper, Module, helpers
 ├─ default_modules.py   # HierarchicalConv2d / Linear (wrap Conv2d/Linear)
 ├─ hierarchical_regularizer.py   # <‑‑ наш движок
 tests/
 └─ test_hierarchical_regularizer.py
 docs/
 └─ hierarchical_regularizer_docs.md
```
* Обёртки слоёв (`HierarchicalConv2d`, `HierarchicalLinear`) знают свои **группы‑метки**: `base_level`, `layer`, `block`, `head`, ...
* В каждом таком слое реализован `get_weights()` → тензор весов `[units, ...]` (первый размер = число фильтров/нейронов).
* `HierarchicalModule.get_param_groups()` собирает рекурсивный список `HierarchicalGroup`, который Regularizer потом фильтрует.

---
## 2. DSL v2 (актуальный)
| Ключ | Допустимые значения | По‑умолчанию | Роль |
|------|--------------------|--------------|------|
| **`type`** | `global\|layerwise\|hierarchical` | — | Что группируем |
| **`groups`** | `null\|base_level\|layer\|blocks\|...` | null | Фильтр по меткам |
| **`norm`** | `sum\|L1\|L2` | `L2` | **Внешняя** норма между группами |
| **`inner_norm`** | `L1\|L2\|L0\|L0approx` | `L2` | **Внутренняя** норма одной группы |
| **`lambda`** | float ≥ 0 | 0.0 | Коэффициент |
| **`children`** | list[Dict] | [] | Разрешено **только** при `type="hierarchical"` |

> `hierarchical` = **контейнер**, он не добавляет штраф сам (его `lambda` игнорируется).

Для болших деталей для этого есть group_loss/readme.md

---
## 3. Реализация L0approx
* Константы в `hierarchical_regularizer.py`:
  ```python
  L0_P = 0.1
  L0_EPS = 1e-8
  ```
* В `_apply_inner` ветка:
  ```python
  elif inner in ("L0", "L0APPROX"):
      return torch.sum(torch.pow(mat.abs() + L0_EPS, L0_P), dim=1)
  ```
* Работает в любом месте, где допустимы другие `inner_norm`.

---
## 4. Юнит‑тесты

См tests/**

---
## 5. Распространённые грабли
* Забыли метку `base_level` в обёртке слоя —> рег‑тор не видит фильтры.
* Добавили `lambda` в `hierarchical` и ждём штрафа. Не будет.
* Путаем `norm` vs `inner_norm`. Внешняя отвечает за sparsity, внутренняя — за норму внутри каждой группы.
* Нормы чувствительны к регистру: приводим к `.upper()`.
