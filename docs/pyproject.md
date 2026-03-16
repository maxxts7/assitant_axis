# `pyproject.toml`

This is the project's PEP 621-compliant configuration file. It defines metadata about the `assistant-axis` Python package -- its name, version, description, required Python version, and runtime dependencies -- as well as the build system used to produce installable artifacts. Everything a tool like `pip` or `hatch` needs to install or build the project is declared here.

---

## `[project]` table

```toml
[project]
```

Opens the standard `[project]` table defined by PEP 621. All core package metadata lives under this heading.

---

```toml
name = "assistant-axis"
```

The distribution name of the package. This is the name used when installing (`pip install assistant-axis`) and the name that would appear on PyPI.

---

```toml
version = "0.1.0"
```

The current version of the package, following semantic versioning. `0.1.0` indicates an early/initial development release.

---

```toml
description = "Tools for computing and steering with the assistant axis"
```

A short, one-line summary of what the project does. Package indexes and `pip show` display this string.

---

```toml
readme = "README.md"
```

Points to the file that contains the long-form project description. Build tools will read `README.md` from the repository root and use its contents as the detailed description (e.g., on a PyPI project page).

---

```toml
requires-python = ">=3.10"
```

Specifies the minimum Python version the project supports. Any Python 3.10 or newer satisfies this constraint; older interpreters will be rejected at install time.

---

```toml
dependencies = [
    "torch>=2.0",
    "transformers>=4.40",
    "accelerate",
    "huggingface_hub",
    "vllm",
    "scikit-learn",
    "numpy",
    "plotly",
    "jupyter",
    "pyarrow",
    "openai",
    "python-dotenv",
    "tqdm",
    "jsonlines",
    "matplotlib>=3.10.8",
]
```

Lists every package that must be installed for the project to work at runtime. When a user runs `pip install assistant-axis`, all of these are pulled in automatically. Each entry is explained below:

| Dependency | Purpose |
|---|---|
| `torch>=2.0` | PyTorch, the deep-learning framework. Version 2.0+ is required (introduces `torch.compile` and other improvements). |
| `transformers>=4.40` | Hugging Face Transformers library for loading and running language models. Version 4.40+ is required. |
| `accelerate` | Hugging Face Accelerate, which simplifies running models across multiple GPUs or mixed-precision setups. No minimum version is pinned. |
| `huggingface_hub` | Client library for downloading models, datasets, and other assets from the Hugging Face Hub. |
| `vllm` | vLLM, a high-throughput inference engine for large language models. |
| `scikit-learn` | Machine-learning utilities (e.g., classifiers, PCA, clustering) used for analysis of model representations. |
| `numpy` | Fundamental array-computing library, a transitive dependency of almost everything above but listed explicitly. |
| `plotly` | Interactive plotting library, likely used for visualising activation or steering results. |
| `jupyter` | The Jupyter ecosystem (notebook server, kernel, etc.) so users can run the project's notebooks. |
| `pyarrow` | Columnar data library, often used for reading/writing Parquet files or working with Hugging Face datasets. |
| `openai` | Official OpenAI Python client, used for calling OpenAI API endpoints (e.g., GPT models). |
| `python-dotenv` | Reads key-value pairs from a `.env` file and sets them as environment variables, useful for API keys. |
| `tqdm` | Provides progress bars for loops and long-running operations. |
| `jsonlines` | Convenience library for reading and writing JSON Lines (`.jsonl`) files. |
| `matplotlib>=3.10.8` | Static plotting library. Version 3.10.8 or newer is required. |

---

## `[project.optional-dependencies]` table

```toml
[project.optional-dependencies]
dev = ["pytest>=7.0"]
```

Defines an optional dependency group called `dev`. Installing with `pip install assistant-axis[dev]` pulls in `pytest` (version 7.0+) for running the test suite. These packages are not installed by default -- only when the `dev` extra is explicitly requested.

---

## `[build-system]` table

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Declares the build system as specified by PEP 517 / PEP 518.

- **`requires`** -- lists the packages that must be available to build the project. Here, only `hatchling` (the Hatch build backend) is needed.
- **`build-backend`** -- tells front-end tools (like `pip` or `build`) which Python object to call to produce wheels and sdists. `hatchling.build` is the entry point provided by the Hatchling package.

---

## `[tool.hatch.build.targets.wheel]` table

```toml
[tool.hatch.build.targets.wheel]
packages = ["assistant_axis"]
```

Hatch-specific configuration that controls what goes into the built wheel. The `packages` key explicitly lists the Python packages (directories with an `__init__.py`) to include. Only the `assistant_axis` directory will be packaged; everything else in the repository (notebooks, docs, scripts, etc.) is excluded from the distributable wheel.
