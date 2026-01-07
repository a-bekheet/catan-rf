"""Setup script for Catan RL research framework."""

from setuptools import setup, find_packages

with open("docs/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="catan-rl",
    version="1.0.0",
    author="Ali Bekheet",
    description="A comprehensive framework for researching RL approaches to Settlers of Catan",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Board Games",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "networkx>=2.6.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "jupyter>=1.0.0",
        "tqdm>=4.60.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "tensorboard>=2.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "isort>=5.0",
            "mypy>=0.900",
            "flake8>=3.8",
        ],
        "analysis": [
            "plotly>=5.0",
            "dash>=2.0",
            "scikit-learn>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "catan-rl-train=scripts.train_dqn_agents:main",
            "catan-rl-evaluate=scripts.evaluate_dqn_agents:main",
            "catan-rl-demo=scripts.demo_integration:main",
            "catan-rl-experiment=examples.run_dqn_experiment:main",
        ],
    },
)