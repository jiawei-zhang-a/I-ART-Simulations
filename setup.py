from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="I-ART",
    version="0.1.0",
    author="Siyu Heng, Jiawei Zhang, and Yang Feng",
    author_email="",
    description="Design-based causal inference is one of the most widely used frameworks for testing causal null hypotheses or inferring about causal parameters from experimental or observational data. The most significant merit of design-based causal inference is that its statistical validity only comes from the study design (e.g., randomization design) and does not require assuming any outcome-generating distributions or models. Although immune to model misspecification, design-based causal inference can still suffer from other data challenges, among which missingness in outcomes is a significant one. However, compared with model-based causal inference, outcome missingness in design-based causal inference is much less studied, largely due to the challenge that design-based causal inference does not assume any outcome distributions/models and, therefore, cannot directly adopt any existing model-based approaches for missing data. To fill this gap, we systematically study the missing outcomes problem in design-based causal inference. First, we use the potential outcomes framework to clarify the minimal assumption (concerning the outcome missingness mechanism) needed for conducting finite-population-exact randomization tests for the null effect (i.e., Fisher's sharp null) and that needed for constructing finite-population-exact confidence sets with missing outcomes. Second, we propose a general framework called \"imputation and re-imputation\" for conducting finite-population-exact randomization tests in design-based causal studies with missing outcomes. Our framework can incorporate any existing outcome imputation algorithms and meanwhile guarantee finite-population-exact type-I error rate control. Third, we extend our framework to conduct covariate adjustment in an exact randomization test with missing outcomes and to construct finite-population-exact confidence sets with missing outcomes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jiawei-zhang-a/I-ART",
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
        'pandas',
        'numpy',
        'scikit-learn',
        'xgboost',
        'statsmodels',
        'lightgbm',
    ],
)