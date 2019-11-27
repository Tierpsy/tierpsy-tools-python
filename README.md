# tierpsytools

tierpsytools is a Python library for dealing with metadata in worm screening experiments and tierpsy feature processing.

## Installation

To install tierpsytools clone the repository from github:

```bash
git clone https://github.com/em812/tierpsytools_python.git
```

Go in the tierpsytools_python directory:

```bash
cd tierpsytools_python
```

and install the package with:

```bash
pip install -e .
```


## Usage

You can import tierpsytools and use modules and functions as in:

```python
import tierpsytools

filtered_features = tierpsytools.filter_features.drop_ventrally_signed(features)
```

or import modules and functions from tierpsytools as in:

```python
from tierpsytools.filter_features import drop_ventrally_signed

filtered_features = drop_ventrally_signed(features)
```
