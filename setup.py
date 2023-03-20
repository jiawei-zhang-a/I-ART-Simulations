from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="project_name",
    version="0.1.0",
    author="",
    author_email="",
    description="post-prediction data imputation framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jiawei-zhang-a/Post-prediction-Causal-Inference",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
    install_requires=[
        'mv-laplace',
        'matplotlib',
        'numpy',
        'pandas',
        'xgboost',
    ],
)
