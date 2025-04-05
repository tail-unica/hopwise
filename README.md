<h1 align="center">🚀 hopwise</h1>

<p align="center">
  <b>RecBole extension with a focus on Knowledge Graphs (KGs) and explainability.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/github/license/tail-unica/hopwise?style=flat-square" />
  <img src="https://img.shields.io/github/repo-size/tail-unica/hopwise?style=flat-square" />
  <img src="https://img.shields.io/github/stars/tail-unica/hopwise?style=flat-square" />
</p>

---

## ✨ Overview

**hopwise** is an advanced extension of the RecBole library, designed to enhance recommendation systems with the power of **knowledge graphs**.
By integrating **knowledge embedding models**, **path-based reasoning methods**, and **path language modeling approaches**, hopwise supports both **recommendation** and **link prediction** tasks with a focus on **explainability**.

---

## 🆕 What's New?

### 🔍 **Added New Path Reasoning Models**
✔️ **PEARLM**
✔️ **KGGLM**
✔️ **PGPR**
✔️ **CAFE**

🛠️ _Future Plans:_ We aim to add **UCPR** (even though it's quite slow ⏳).

### 🧩 **Added 14 Knowledge Graph Embedding Models**
✔️ **TransE**
✔️ **TransH**
✔️ **TransD**
✔️ **TransR**
✔️ **TorusE**
✔️ **ComplEx**
✔️ **Analogy**
✔️ **TuckER**
✔️ **RESCAL**
✔️ **DistMult**
✔️ **ConvE**
✔️ **ConvKB**
✔️ **RotatE**
✔️ **HolE**

---

## ⚡ Installation

To install the project, you need to use `uv`. Follow the steps below to set up the environment and install the necessary dependencies.

### 🔹 Prerequisites
- ✅ Python **3.9**, **3.10**, or **3.11**
- ✅ [`uv`](https://github.com/astral-sh/uv) package manager

### 🔹 Steps

1️⃣ **Clone the repository**
```sh
git clone https://github.com/tail-unica/hopwise.git
cd hopwise
```
2️⃣ Install **uv** and create a virtual environment.<br>
We suggest installing **uv** as a [standalone application](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) instead of using pip to avoid issues and let **uv** create a dedicated virtual environment.<br>
Once installed, create the virtual environment
```sh
uv venv --python PYTHON_VERSION --prompt hopwise
```
`PYTHON_VERSION` must be one of 3.9, 3.10, 3.11, while `--prompt hopwise` customizes the virtual environment name that appears on the shell.

3️⃣ Install project dependencies
```sh
uv sync
```
Some models require extra dependencies.
Check out pyproject.toml for optional dependencies.
For example, to install NNCF:
```sh
uv sync --extra nncf
```

> 📢 **Windows:** For proper DGL installation, please follow the [official DGL installation guide](https://www.dgl.ai/pages/start.html). Windows builds may encounter DLL linking issues with standard installation methods. Pre-built packages from the official source are recommended. Otherwise, using the Windows Subsystem for Linux (WSL) might be feasible as a solution.

🚀 Usage

Run the project with the following command:
```sh
uv run run_hopwise.py --model MODEL --dataset DATASET --config_files CONF_FILE_1.yaml CONF_FILE_2.yaml
```

Override config parameters directly from the CLI using =:
```sh
uv run run_hopwise.py --epochs=20
```

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.