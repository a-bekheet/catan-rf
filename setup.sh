#!/bin/bash
# Catan Multi-Agent RL Setup Script
# This script automatically sets up the environment using UV

set -e  # Exit on error

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Catan Multi-Agent RL Training System Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "📦 UV not found. Installing UV..."
    echo ""

    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            echo "   Using Homebrew to install UV..."
            brew install uv
        else
            echo "   Using curl to install UV..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "   Using curl to install UV..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        echo "   Unsupported OS. Please install UV manually:"
        echo "   https://github.com/astral-sh/uv"
        exit 1
    fi

    # Add UV to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"

    echo ""
    echo "✓ UV installed successfully!"
else
    echo "✓ UV is already installed (version: $(uv --version))"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📚 Installing Dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This will install:"
echo "  • PyTorch (Deep Learning)"
echo "  • Ray RLlib (PPO Agent)"
echo "  • TorchRL (SAC Agent)"
echo "  • LangChain/LangGraph (LLM Agent)"
echo "  • FastAPI (Web Interface)"
echo "  • Rich (Beautiful CLI)"
echo "  • And more..."
echo ""

# Install dependencies with UV
echo "Installing with UV (this is 10-100x faster than pip)..."
uv sync

echo ""
echo "✓ All dependencies installed!"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate the virtual environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Start training AI agents:"
echo "     python train_agents.py"
echo ""
echo "  3. Or use the interactive menu:"
echo "     python run.py"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📖 Documentation:"
echo "   • Quick Start: QUICK_START_GUIDE.md"
echo "   • UV Guide: UV_SETUP.md"
echo "   • Full Docs: MULTI_AGENT_TRAINING.md"
echo ""
echo "🌟 Optional: Set API keys for LLM agent"
echo "   export OPENAI_API_KEY='your-key-here'"
echo "   export ANTHROPIC_API_KEY='your-key-here'"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
