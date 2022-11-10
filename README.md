

<h1 align="center">
<img src="[https://github.com/DanielPuentee/outdpik/blob/main/branding/logo/primary/outdpik.png?raw=true](https://github.com/ikumpli/A-deep-learning-based-approach-for-lithium-ion-battery-RUL-prediction-based-on-data-augmentation/blob/main/logo.png)" width="300">
</h1><br>

# INDUSTRY CHALLENGE

[![PyPI Latest Release](https://img.shields.io/pypi/v/outdpik.svg)](https://pypi.org/project/outdpik/)
[![PyPI License](https://img.shields.io/pypi/l/outdpik.svg)](license.txt)
[![Package Status](https://img.shields.io/pypi/status/pandas.svg)](https://pypi.org/project/outdpik/)
[![Documentation Status](https://readthedocs.org/projects/outdpik/badge/?version=latest)](https://outdpik.readthedocs.io/en/latest/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## What is it?
Outdpik is an open source Python package that provides different methods for outlier detection. 
It aims to be the fundamental high-level package for this purpose. 
Additionally, it offers visualization methods for the outlier analysis.

## Main Features
Here are just a few of the things that outdpik does well:

- It supports numpy arrays and pandas dataframes
- Multiple outlier detection techniques that can be combined
- Powerful visualizations
- Flexible at including one or more columns for the analysis

## Where to get it
The source code is currently hosted on GitHub at:
https://github.com/DanielPuentee/outdpik

Installer for the latest released version is available at the [Python
Package Index (PyPI)](https://pypi.org/project/outdpik)

```sh
# PyPI
pip install outdpik
```

## How to use outdpik
Examples of configuring and running outpdik:

```python
import outpdik as outdp
outdp = outdp()
```

We proceed to detect outliers returning a dictionary of numeric features and the outliers instances:

```python
outliers_dict = outdp.outliers(df = df, cols = "all")
```
Plotting advantages:

```python
outdp.plot_outliers(df = df, col = "x")
```
<h1 align="center">
<img src="https://github.com/DanielPuentee/outdpik/blob/main/branding/logo/primary/graph.png?raw=true" width=450 alt="Strip plot outliers detection">
</h1><br>

## Dependencies
- [pandas - Provides fast, flexible, and expressive data structures designed to make working with "relational" or "labeled" data both easy and intuitive](https://pandas.pydata.org/)
- [NumPy - Adds support for large, multi-dimensional arrays, matrices and high-level mathematical functions to operate on these arrays](https://www.numpy.org)
- [SciPy - Includes modules for statistics, optimization, integration, linear algebra, Fourier transforms, signal and image processing, ODE solvers, and more](https://scipy.org/)
- [matplotlib - Comprehensive library for creating static, animated, and interactive visualizations in Python](https://matplotlib.org/)
- [seaborn - Provides a high-level interface for drawing attractive statistical graphics](https://seaborn.pydata.org/)

## License
This project is licensed under the terms of the [GNU](https://github.com/DanielPuentee/outdpik/blob/main/license.txt) - see the LICENSE file for details.

## Documentation
The official documentation is hosted on: https://outdpik.readthedocs.io/en/latest/

## Development
Want to contribute? Great!
Open a discussion in Github in this repo and we will answer as soon as possible.
