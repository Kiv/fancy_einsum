# Fancy Einsum

This is a simple wrapper around `np.einsum` and `torch.einsum` that allows the use of self-documenting variable names instead of just single letters in the equations. Inspired by the syntax in [einops](https://github.com/arogozhnikov/einops).

For example, instead of writing:

```python
import torch
torch.einsum('bct,bcs->bcts', a, b)
```

or 

```python
import numpy as np
np.einsum('bct,bcs->bcts', a, b)
```

With this library you can write:

```python
from fancy_einsum import einsum
einsum('batch channels time1, batch channels time2 -> batch channels time1 time2', a, b)
```