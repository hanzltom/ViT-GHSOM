# Self-Organizing Maps and Their Modern Extensions thesis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains all the necessary attachments associated with the thesis ```Self-Organizing Maps and Their Modern Extensions```. It includes the source code for all implementations of the models,
experiments in jupyter notebooks, assignment, declaration, and the thesis document itself.

## Structure:

```text
ViT-GSOM_thesis/
├── models/
│   ├── data/                                       # Shared datasets storage
│   ├── GHSOM/                                      # Growing Hierarchical Self-Organizing Map
│   │   ├── data_extraction.ipynb                   # TIME Magazine extraction 
│   │   ├── GHSOM_iris.ipynb                        # GHSOM on IRIS dataset
│   │   ├── GHSOM_Times.ipynb                       # GHSOM on TIME Magazine dataset
│   │   ├── GHSOM.py                                # Core GHSOM implementation
│   │   ├── GSOM.py                                 # Base GSOM implementation
│   │   └── help_functions.py                       
│   ├── SOM/                                        # Standard Self-Organizing Map
│   │   ├── SOM_training_IRIS.ipynb                 # SOM on IRIS dataset
│   │   ├── SOM.py                                  # Core implementation
│   │   └── help_functions.py                       
│   ├── ViT-GSOM/                                   # Vision Transformer Growing Self-Organizing Map
│   │   ├── data/                                   # ViT-GSOM - specific data
│   │   ├── ViT-GSOM_training_fashionMNIST.ipynb    # ViT-GSOM on FashionMNIST dataset
│   │   ├── ViT-GSOM_training_MNIST.ipynb           # ViT-GSOM on MNIST dataset
│   │   ├── ViT-GSOM_training_USPS_infinite.ipynb   # ViT-GSOM on USPS dataset infinite training
│   │   ├── ViT-GSOM_training_USPS.ipynb            # ViT-GSOM on USPS dataset
│   │   ├── ViT-GSOM_tuning_USPS.ipynb              # ViT-GSOM on USPS dataset hyperparameter tuning
│   │   ├── ViTGSOM.py                              # Core ViT-GSOM implementation
│   │   └── help_functions.py                       
│   └── ViT-SOM/                                    # Vision Transformer Self-Organizing Map
│       ├── ViT-SOM_training_fashionMNIST.ipynb     # ViT-SOM on FashionMNIST dataset
│       ├── ViT-SOM_training_MNIST.ipynb            # ViT-SOM on MNIST dataset
│       ├── ViT-SOM_training_USPS.ipynb             # ViT-GSOM on USPS dataset
│       ├── ViTSOM.py                               # Core ViT-SOM implementation
│       └── help_functions.py                       
├── text/
│   └── thesis.pdf                                  # Thesis text
├── assignment.pdf
├── declaration_ai.pdf
├── LICENSE                                         # Project license
├── README.md                                       
└── requirements.txt                                # Python dependencies
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hanzltom/ViT-GSOM_thesis.git
   cd ViT-GSOM_thesis
    ```
   
2. **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
   
3. **Install dependencies:**
    ```bash
   pip install -r requirements.txt
    ```

---

## Additional information

- **Author**: Tomáš Hanzlík
- **Username**: hanzlto3
- **E-mail**: hanzlto3@fit.cvut.cz
- **Date**: 11.05.2026



<img src="https://fit.cvut.cz/static/images/fit-cvut-logo-en.svg" alt="FIT CTU logo" height="200">

This software was developed with the support of the **Faculty of Information Technology, Czech Technical University in Prague**.
For more information, visit [fit.cvut.cz](https://fit.cvut.cz).
