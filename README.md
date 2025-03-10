<h1 align="center">
    hopwise
</h1>


<p align="center">
🚀 Recbole extension with focus on Knowledge Graphs (KGs) and interpretability/explainability.
</p>

---
_hopwise_ is an advanced extension of the Recbole library, designed to enhance recommendation systems with the power of knowledge graphs. By integrating knowledge embedding models, path-based reasoning methods, and path language modeling approaches, hopwise supports both recommendation and link prediction tasks with a focus on interpretability and self-explanation.

## Installation

To install the project, you need to use `uv`. Follow the steps below to set up the environment and install the necessary dependencies.

### Prerequisites

- Python 3.9, 3.10, or 3.11
- `uv` package manager

### Steps

1. **Clone the repository:**

    ```sh
    git clone https://github.com/tail-unica/hopwise.git
    cd hopwise
    ```

2. **Install `uv` and set the Python version:**

    ```sh
    pip install uv
    uv set python-version 3.9  # or 3.10, 3.11
    ```

3. **Install the project dependencies:**

    ```sh
    uv sync
    ```

4. **Some models require extra dependencies. Check out [pyproject.toml](pyproject.toml) if the model is included under `[project.optional-dependencies]`. For instance, to install NNCF:**

    ```sh
    uv sync --extra nncf
    ```

## Usage

To run the project, you can use the following command:

```sh
uv run run_hopwise.py --model MODEL --dataset DATASET --config_files CONF_FILE_1.yaml CONF_FILE_2.yaml
```

You can also ovveride config parameters directly from the cli (`=` is used to separate name and value):

```sh
uv run run_hopwise.py --epochs=20
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.