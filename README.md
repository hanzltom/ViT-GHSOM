# Thesis's attachments README

This repository contains all the necessary attachments associated with the thesis. It includes the source code for all implementations of the models,
experiments in ipynb files, visualizations, Latex source code for the thesis and the thesis document itself.

Structure:

```text
ViT-GSOM_thesis/
├── data/                       # Global dataset storage
├── thesis/...                  # Latex source code for the thesis
├── GHSOM/                      # Growing Hierarchical Self-Organizing Map
│   ├── data_extraction.ipynb
│   ├── GHSOM_iris.ipynb
│   ├── GHSOM_Times.ipynb
│   ├── GHSOM.py                # Core implementation
│   ├── GSOM.py                 # Base GSOM implementation
│   └── help_functions.py
├── SOM/                        # Standard Self-Organizing Map
│   ├── SOM_training_IRIS.ipynb
│   ├── SOM.py                  # Core implementation
│   └── help_functions.py
├── ViT-GSOM/                   # Vision Transformer Growing Self-Organizing Map
│   ├── data/                   # ViT-GSOM - specific data
│   ├── ViT-GSOM_training_fashionMNIST.ipynb
│   ├── ViT-GSOM_training_MNIST.ipynb
│   ├── ViT-GSOM_training_USPS_infinite.ipynb
│   ├── ViT-GSOM_training_USPS.ipynb
│   ├── ViT-GSOM_tuning_USPS.ipynb
│   ├── ViTGSOM.py              # Core implementation
│   └── help_functions.py
├── ViT-SOM/                    # Vision Transformer Self-Organizing Map
│   ├── data/                   # ViT-SOM - specific data
│   ├── ViT-SOM_training_fashionMNIST.ipynb
│   ├── ViT-SOM_training_MNIST.ipynb
│   ├── ViT-SOM_training_USPS.ipynb
│   ├── ViTSOM.py               # Core implementation
│   └── help_functions.py
├── LICENSE                     # Project license
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

## Additional information

- **Author**: Tomáš Hanzlík
- **Username**: hanzlto3
- **E-mail**: hanzlto3@fit.cvut.cz
- **Date**: 19.04.2026



<img src="https://fit.cvut.cz/static/images/fit-cvut-logo-en.svg" alt="FIT CTU logo" height="200">

This software was developed with the support of the **Faculty of Information Technology, Czech Technical University in Prague**.
For more information, visit [fit.cvut.cz](https://fit.cvut.cz).