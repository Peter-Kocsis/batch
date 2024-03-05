<div align="center">

<h1 align="center">Batch</h1>


Generic python module for handling dictionary-based batch data. 
______________________________________________________________________

[![PyPI - Python Version](https://img.shields.io/badge/python-3.8_|_3.9_|_3.10_|_3.11_|3.12-blue)](https://pypi.org/project/batch/)
[![PyPI Status](https://img.shields.io/badge/pip-v0.1-green)](https://pypi.org/project/batch/)

<!--
[![CodeFactor](https://www.codefactor.io/repository/github/Lightning-AI/lightning/badge)](https://www.codefactor.io/repository/github/Lightning-AI/lightning)
-->

</div>

# Purpose
Are you working with data of similar modalities, and often have to apply the same function to multiple elements? 
Are you using something similar to this:

```python
batch = {
    "image_a": image_a,
    "image_b": image_b,
    "image_c": image_c
}

# Move to another device
for key in batch:
    batch[key] = batch[key].to(device)

# Transform
for key in batch:
    batch[key] = batch[key] * 2 + 1
    
# Combine
for key in batch:
    batch[key] = batch[key] + batch_2[key]

# Process
for key in batch:
    batch[key] = batch[key].max()
```
If the answer is yes, **then this module is for you!**

Our ***Batch*** package is a generic wrapper for dictionary-based batch data. 
It provides a simple way to apply the same function or operator to the whole batch. 
The module is completely device and container independent, you can use it with PyTorch, NumPy or any other libraries.

```python
batch = Batch(
    image_a=image_a, 
    image_b=image_b, 
    image_c=image_c)

# Move to another device
batch = batch.to(device)

# Transform
batch = batch * 2 + 1

# Combine
batch = batch + batch_2

# Process
batch = batch.max()
```

# Installation
```
pip install batch
```

# Usage
The example below demonstrates a few basic use-cases using NumPy. Similarly, PyTorch or other containers can also be used.  

## Import

```python
from batch import Batch
```

## Instantiation
### Direct
```python
# Create a batch directly
batch = Batch(
    image_a=np.random.rand(256, 256, 3), 
    image_b=np.random.rand(256, 256, 3), 
    image_c=np.random.rand(256, 256, 3))
```

### From dictionary
```python
batch = {
    "image_a": np.random.rand(256, 256, 3), 
    "image_b": np.random.rand(256, 256, 3), 
    "image_c": np.random.rand(256, 256, 3)}
# Create a batch from a dictionary
batch = Batch.from_dict(batch)
```

### From tensor
```python
image = np.random.rand(256, 256, 9)
    
# Create a batch from a tensor by splitting the tensor along one dimension and store the splits
data_splits = {
    "image_a": 1, 
    "image_b": 1, 
    "image_c": 1}
dim = 2
batch = Batch.from_tensor(batch, data_splits, dim=dim)
```

## Indexing
A Batch is a string-keyed dictionary, with potentially mapping or iterable values. 

### String index
When a string index is given, then it is always interpreted as a key. 

#### Single key
Querying a single key returns the value associated:
```python
image_a = batch["image_a"]
```
You can even index deeper using `.` as separator:
```python
batch_2 = Batch(input=batch)
image_a = batch_2["input.image_a"]
```

#### Multiple keys
Querying multiple keys (tuple or list) return a new batch with the selected keys:
```python
batch_out = batch["image_a", "image_b"]
```

#### Wildcard query
Wildcard query is also supported and returns a new batch with the matching keys:
```python
batch_out = batch.query_wildcard("image_*")
```
### Integer index
When an integer index is given, then it is always interpreted as an index to the elements and returns a new batch with the indexed elements: 
```python
batch_out = batch[:,:,0]
```


## Processing a batch
### Operators
You can use the followingunary, binary and reverse operators: 
```python
# Unary operators
"__not__", "__abs__", "__index__", "__inv__", "__invert__", "__neg__", "__pos__",

# Binary operators
"__add__", "__and__", "__concat__", "__floordiv__", "__lshift__", "__mod__", "__mul__",
"__or__", "__pow__", "__rshift__", "__sub__", "__truediv__", "__xor__", "__eq__",

# Reverse operators
"__radd__", "__rand__", "__rmul__",
"__ror__", "__rsub__",  "__rxor__",

# In-place operators
"__iadd__", "__iand__", "__iconcat__", "__ifloordiv__", "__ilshift__", "__imod__", "__imul__",
"__ior__", "__ipow__", "__irshift__", "__isub__", "__itruediv__", "__ixor__"
```
Example:
```python
# Use operators
batch_out = batch + batch_2 * 2
```

### Member functions
You can use any member functions of the underlying container, for example:
```python
# Use member functions
batch_out = batch.mean(axis=2)
```

### Map
You can easily apply a function to the whole batch:
```python
batch = batch.map(list)  # Converts all elements to list
batch = batch.map(np.stack, axis=0)  # Concatenates all elements to a single tensor
```

### Map keys
You can also apply a function to the keys:
```python
batch = batch.map_keys(lambda x: f"{x}_2")  # Add suffix
```


# Limitations
A few limitations to consider when using this module:
* Use only string keys for the batch.
* Don't use keys starting with underscore (`_`).
* Slice indexing is not implemented yet.
* Generic iterable indexing is not implemented, only tuple and list.
* Code documentation is in progress. 
* Some features are not yet documented here, please refer to the code directly. 

If you have any ideas or requests, feel free to open an issue or a pull request.


