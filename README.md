# Gitor_plus

Official repository for the extended version of **Gitor**, a graph-based code clone detection approach that builds a **global sample graph** from source-code-level information and then performs clone detection using either **cosine similarity** or a **classifier-based method**.

This repository corresponds to the **journal extension** of our previous FSE 2023 paper. Compared with the conference version, this version adds:

- code metrics as a new source of individual information;
- grouped experiments on multiple metric combinations;
- a classifier-based detection setting with fully connected layers;
- broader comparative evaluation against traditional, deep learning-based, and LLM-based baselines;
- additional scalability experiments.

## Overview

Code clone detection aims to identify similar code fragments and is important for software maintenance and evolution. Existing techniques and tools for code clone detection have achieved encouraging results, but most of them analyze code samples individually and do not explicitly exploit the underlying connections among different code samples. In this work, we propose **Gitor** to capture such connections for clone detection.

Specifically, given a source code database, we first tokenize all code samples to extract pre-defined individual information (**keywords** and **code metrics**). We then construct a large global sample graph in which each node denotes either a code sample or a type of individual information. Based on this graph, we apply node embedding to obtain vector representations of all code samples. Subsequently, we either directly compute cosine similarity to efficiently detect potential clone pairs or input these vectors into a neural network classifier to identify more complex semantic clones. In our experiments on BigCloneBench, Gitor achieves highly competitive effectiveness and scalability compared with representative traditional and deep learning-based clone detectors.

## Repository Structure

```text
Gitor_plus/
├─ README.md
├─ data/
│  └─ processed/
│     ├─ BCB/                   # processed clone-pair csv files
│     ├─ filtered_code_files/   # filtered Java source files
│     ├─ id2sourcecode/         # function-id to source-code mapping
│     └─ split_data/            # train/val/test split files
├─ results/
│  ├─ figures/                  # generated figures used in the paper
│  └─ tables/                   # generated tables used in the paper
└─ scripts/
   ├─ extract.py
   ├─ metrics.py
   ├─ train.py
   ├─ train_fcl.py
   ├─ detect.py
   ├─ train_sca.py
   ├─ train_sca_fcl.py
   ├─ detect_sca_1.py
   ├─ detect_sca_2.py
   ├─ get_vector.py
   ├─ get_figure.py
   ├─ get_figute6.py
   ├─ get_txt.py
   └─ get_sca_id2sourcecode.py
```

## Main Scripts

- `scripts/extract.py`  
  Extracts Java reserved keywords from source code and builds the keyword graph.

- `scripts/metrics.py`  
  Extracts code metrics and builds the metric graph.

- `scripts/train.py`  
  Builds the global graph and learns node embeddings with ProNE.

- `scripts/detect.py`  
  Performs cosine-similarity-based clone detection and reports precision, recall, and F1.

- `scripts/train_fcl.py`  
  Trains and evaluates the classifier-based variants of Gitor.

- `scripts/train_sca.py`, `scripts/train_sca_fcl.py`, `scripts/detect_sca_1.py`, `scripts/detect_sca_2.py`  
  Scripts used for the scalability experiments.

- `scripts/get_figure.py`, `scripts/get_figute6.py`, `scripts/get_txt.py`, `scripts/get_vector.py`, `scripts/get_sca_id2sourcecode.py`  
  Utility scripts for generating intermediate artifacts and paper-ready outputs.

## Environment Setup

The current implementation uses Python and depends mainly on the following packages:

- `torch`
- `pandas`
- `numpy`
- `networkx`
- `nodevectors`
- `javalang`
- `scikit-learn`
- `gensim`

A typical setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

GPU is recommended for the classifier-based experiments, but the cosine-similarity-based pipeline can also be run on CPU.

## Step-by-Step Reproduction Guide

The scripts in this repository are organized around the main experimental workflow of the paper.

### 1. Build graph embeddings

Train graph embeddings with the desired embedding dimension:

```bash
python scripts/train.py --embed_dim 64
```

This step builds the global graph from the processed source-code artifacts and learns node embeddings with ProNE.

### 2. Run cosine-similarity-based detection

After embeddings are generated, use:

```bash
python scripts/detect.py
```

This reproduces the cosine-similarity-based detection setting used in the paper.

### 3. Run the classifier-based setting

To train and evaluate the fully connected classifier variants:

```bash
python scripts/train_fcl.py
```

This script evaluates the classifier-based variants across multiple embedding dimensions and reports accuracy, precision, recall, and F1.

### 4. Run scalability experiments

To reproduce the scalability experiments:

```bash
python scripts/train_sca.py
python scripts/train_sca_fcl.py
python scripts/detect_sca_1.py
python scripts/detect_sca_2.py
```

### 5. Generate figures and tables

Use the utility scripts to generate paper-ready outputs:

```bash
python scripts/get_figure.py
python scripts/get_txt.py
python scripts/get_vector.py
python scripts/get_sca_id2sourcecode.py
```

Generated figures and tables are stored under:

- `results/figures/`
- `results/tables/`

## Relation to the FSE 2023 Version

The original FSE 2023 repository focused on the conference version of Gitor, which mainly used:

- keywords as the main individual information source;
- side information from the conference version;
- cosine-similarity-based clone detection.

This repository extends that version by adding:

- 27 code metrics;
- combined keyword + metric graph construction;
- classifier-based detection with fully connected layers;
- expanded evaluation and scalability experiments.

## Citation

If you find this repository useful, please cite the journal version of this work. We will update the BibTeX entry here after the paper is accepted and published.

```bibtex
% Placeholder: the journal citation will be added after publication.
```
