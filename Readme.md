# Micro RF project

## Env setup

Python3 and pip are required

1. Create venv

```bash
python3 -m venv ${envName}
```

Activate venv

```bash
source myenv/bin/${envName}
```

Install packages from requirements.txt

```bash
pip3 install -r requirements.txt
```

2. Run script

This script accepts two parameters:

```
map that accepts values 4 (4x4 map) and 8 (8x8 map)
```

```
mode that accepts values train and test
```

To run the script invoke main.py with parameters for example:

```bash
python3 main.py 4 train
```
