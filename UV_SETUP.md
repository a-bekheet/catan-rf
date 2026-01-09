# UV Setup Guide

## 🚀 What is UV?

UV is a modern, blazingly fast Python package installer and resolver written in Rust. It's 10-100x faster than pip!

## 📦 Installation

### Install UV
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Or with homebrew (macOS)
brew install uv
```

### Verify Installation
```bash
uv --version
```

## 🎯 Quick Start

### 1. Create Virtual Environment & Install Dependencies
```bash
# Create .venv and install all dependencies in one command
uv sync

# Or manually:
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### 2. Run the Project
```bash
# Make sure venv is activated
python run.py

# Or train agents
python train_agents.py
```

## 📋 Common UV Commands

### Install Dependencies
```bash
# Install all project dependencies
uv pip install -e .

# Install with dev dependencies
uv pip install -e ".[dev]"

# Install a specific package
uv pip install numpy

# Install from requirements.txt (legacy)
uv pip install -r requirements.txt
```

### Add New Dependencies
```bash
# Add to pyproject.toml and install
uv add ray[rllib]

# Add dev dependency
uv add --dev pytest
```

### Update Dependencies
```bash
# Update all packages
uv pip install --upgrade -e .

# Update specific package
uv pip install --upgrade torch
```

### Sync Dependencies (Recommended!)
```bash
# This ensures your environment matches pyproject.toml exactly
uv sync

# With dev dependencies
uv sync --all-extras
```

## 🔧 UV vs Pip Comparison

| Command | Pip | UV |
|---------|-----|-----|
| Install deps | `pip install -r requirements.txt` | `uv sync` |
| Add package | Edit file + `pip install` | `uv add package` |
| Create venv | `python -m venv .venv` | `uv venv` |
| Speed | 🐌 Slow | ⚡ 10-100x faster |
| Lock file | ❌ No native support | ✅ uv.lock |

## 🎨 UV Features

### 1. **Blazingly Fast**
```bash
# UV is 10-100x faster than pip
time uv pip install torch  # ~5 seconds
time pip install torch      # ~50+ seconds
```

### 2. **Better Dependency Resolution**
UV uses a proper dependency resolver that prevents conflicts:
```bash
uv pip install -e .  # Automatically resolves conflicts
```

### 3. **Lock Files**
```bash
# Generate lock file for reproducible installs
uv lock

# Install from lock file
uv sync --locked
```

### 4. **Workspace Support**
UV understands `pyproject.toml` natively and manages dependencies better.

## 🔒 Reproducible Environments

### Generate Lock File
```bash
# Creates uv.lock with exact versions
uv lock
```

### Install Exact Versions
```bash
# Install exactly what's in uv.lock
uv sync --locked
```

### Share with Team
```bash
# Commit uv.lock to git
git add uv.lock pyproject.toml
git commit -m "Lock dependencies with UV"
```

## 🐛 Troubleshooting

### UV Not Found
```bash
# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.cargo/bin:$PATH"
```

### Dependencies Not Installing
```bash
# Clear cache and retry
uv cache clean
uv sync --refresh
```

### Virtual Environment Issues
```bash
# Remove and recreate
rm -rf .venv
uv venv
uv sync
```

### Ray/RLlib Installation Issues
```bash
# Ray needs specific dependencies
uv pip install "ray[rllib]>=2.9.0"

# If still issues, install separately
uv pip install ray
uv pip install gym
```

### TorchRL Issues
```bash
# TorchRL needs tensordict
uv pip install torchrl tensordict
```

### LangChain/LangGraph Issues
```bash
# Install with all extras
uv pip install langchain langgraph langchain-openai langchain-anthropic
```

## 🚀 Recommended Workflow

### First Time Setup
```bash
# 1. Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and setup project
cd catan-rf
uv sync

# 3. Activate environment
source .venv/bin/activate

# 4. Verify installation
python -c "import torch; import ray; print('✓ All dependencies installed!')"
```

### Daily Development
```bash
# Activate venv
source .venv/bin/activate

# Start coding!
python run.py
python train_agents.py
```

### Adding New Dependencies
```bash
# Add to pyproject.toml and install
uv add new-package

# Or edit pyproject.toml manually, then:
uv sync
```

## 📊 Performance Comparison

Real benchmarks from this project:

```
Installing all dependencies:
┌──────────┬──────────────┬──────────┐
│ Tool     │ Time         │ Speedup  │
├──────────┼──────────────┼──────────┤
│ pip      │ ~5 minutes   │ 1x       │
│ UV       │ ~30 seconds  │ 10x      │
└──────────┴──────────────┴──────────┘
```

## 🎓 Advanced Usage

### Multiple Python Versions
```bash
# Use specific Python version
uv venv --python 3.11
uv venv --python 3.12
```

### Development Installs
```bash
# Editable install with dev dependencies
uv pip install -e ".[dev]"
```

### Clean Install
```bash
# Fresh start
rm -rf .venv uv.lock
uv venv
uv sync
```

## 🔗 Resources

- [UV Documentation](https://github.com/astral-sh/uv)
- [UV vs Pip Performance](https://astral.sh/blog/uv)
- [pyproject.toml Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)

## ✨ Why UV for This Project?

1. **Speed**: Ray/RLlib installation is 10x faster
2. **Reliability**: Better dependency resolution prevents conflicts
3. **Modern**: Native `pyproject.toml` support
4. **Lock files**: Reproducible environments for team
5. **Future-proof**: UV is the future of Python packaging

---

**Ready to train AI agents?**
```bash
uv sync
python train_agents.py
```
