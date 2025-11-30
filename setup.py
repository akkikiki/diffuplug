"""
Setup script for dllm_plugin - vLLM plugin for diffusion language models.
"""

from setuptools import setup, find_packages

setup(
    name="dllm_plugin",
    version="0.1.0",
    description="vLLM plugin for Diffusion Language Models (Dream and LLaDA)",
    author="dllm_plugin contributors",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "vllm>=0.6.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-mock>=3.10.0",
        ],
    },
    entry_points={
        "vllm.general_plugins": [
            "register_dllm_models = dllm_plugin:register"
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
