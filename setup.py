from setuptools import setup, find_packages

setup(
    name="hierarchical-monitoring",
    version="0.1.0",
    description="Soft-gating architecture for hierarchical monitoring systems",
    author="Carlo Leonardo Attubato",
    author_email="carlo.attubato@gmail.com",
    url="https://github.com/carlo-leonardo-attubato/hierarchical-monitoring",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "wandb>=0.12.0",
        "hydra-core>=1.2.0",
        "omegaconf>=2.2.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "pytest>=6.2.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
