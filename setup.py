from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vlm_benchmark",
    version="0.1.0",
    author="VLM Benchmark Team",
    author_email="your-email@example.com",
    description="A comprehensive benchmark system for Vision-Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vlm-benchmark",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    # extras_require={
    #     "dev": [
    #         "pytest>=6.0",
    #         "pytest-asyncio>=0.21.0",
    #         "black>=22.0",
    #         "flake8>=4.0",
    #         "mypy>=0.950",
    #     ],
    #     "full": [
    #         "transformers>=4.30.0",
    #         "torch>=2.0.0",
    #         "torchvision>=0.15.0",
    #         "openai>=1.0.0",
    #         "nltk>=3.8",
    #         "rouge-score>=0.1.2",
    #     ]
    # },
    entry_points={
        "console_scripts": [
            "vlm-benchmark=vlm_benchmark.cli:main",
        ],
    },
)