# tierpsytools

tierpsytools is a Python library for dealing with metadata in worm screening experiments and tierpsy feature processing.

## Installation
It is recommended to create an environmemt to install tierpsytools in. In conda you can create an environmnet called tierpsytools with:

```bash
conda create -n tierpsytools
```

Activate the environment with:

```bash
source activate tierpsytools
```

To install tierpsytools clone the repository from github:

```bash
git clone https://github.com/Tierpsy/tierpsy-tools-python.git
```

Go in the tierpsy-tools-python directory:

```bash
cd tierpsy-tools-python
```

Firt, install the dependences with:

```bash
conda install --file requirements.txt
pip install imgstore
```

Then install the tierpsytools package with:

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
