# Lithium-ion battery optimal RUL prediction combining LSTM and GANs
[![Package Status](https://img.shields.io/pypi/status/pandas.svg)](https://pypi.org/project/outdpik/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<img src="./images/intro.gif" alt="alt text" width="1000">

This repository contains the code used for the research study of RUL prediction, based on data augmentation .
## What is this project about?
<br>
For this project, based on the RUL prediction, a deep learning based approach trained on the widely-used
Oxford battery degradation dataset with the help of generative adversarial
networks (GANS) has been implemented.

Lithium-ion batteries are one of the most widely used solutions in many
sectors, such as electric vehicles, thanks to their higher energy density and
low self-discharge. With the use and passage of time, batteries degrade and
eventually die, endangering the integrity of the objects they power.

To prevent all these from happening a “A deep learning based approach for lithium-ion-battery RUL
prediction based on data augmentation” model has been designed as our project.

## Algorithms used
- [x] Simple LSTM & GRU
- [x] Bidirectional LSTM & GRU
- [x] LSTM-GANs

## Folder distribution

```
  .
  ├── 01_dev                    
  │   ├── functions         # Functions used 
  │   ├── hyperas_tunning         # Neural Network tunning notebook
  │   └── ...                # Rest of the notebooks used for the project development
  ├── images
  └── ...
```

## Development 👋
<br>
Want to contribute? Great!
Open a discussion in Github in this repo and we will answer as soon as possible.

## Authors

- Jon Amelibia
- Iker Cumplido
- Aitor Hernandez
- Daniel Puente
- Iñigo Ugarte
