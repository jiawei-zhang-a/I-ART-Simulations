# RIMO  
In many practical studies, among the outcomes Y1, . . . , YK , one or multiple of them are
incomplete. In randomized experiments, simply ignoring the missing outcomes may lead to
statistical inference not finite population exact anymore. In matched observational studies,
people routinely exclude study subjects with missing outcomes, which would substantially
the statistical power due to discarding many study samples.
These framework is for testing post-prediction missing data imputation



## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [How to Contribute](#how-to-contribute)
- [License](#license)

## Installation

[Installation instructions]

## Usage


## How to Contribute

We welcome contributions to the `rimo` package! Here's how you can contribute:

### 1. Fork and Clone the Repository:

Fork the main repository on GitHub and then clone your fork locally.

### 2. Create a New Branch:

Create a new branch for your feature or bugfix:

```bash
git checkout -b my-feature-branch
```


3. **Make Your Changes:**  
Make the necessary changes in your local environment.

4. **Build the Package:**  
Navigate to the package directory (`rimo/`) and run:

```bash
python setup.py sdist bdist_wheel
```

This will create a `dist` directory containing the built package.

5. **Test the Package Locally:**  
Before pushing your changes, test the package locally:
  - Install the package from the local distribution files:
    ```
    pip install ./dist/rimo-0.1-py3-none-any.whl
    ```
  - Test the package in a Python environment:
    ```python
    import rimo
    # Use functions or classes from your package
    result = rimo.retrain_test(...)
    ```

6. **Using a Virtual Environment (Recommended):**  
  - Create a new virtual environment:
    ```
    python -m venv rimo_test_env
    ```
  - Activate the virtual environment:
    - On macOS and Linux:
      ```
      source rimo_test_env/bin/activate
      ```
    - On Windows:
      ```
      .\rimo_test_env\Scripts\activate
      ```
  - Install your package in the virtual environment:
    ```
    pip install ./dist/rimo-0.1-py3-none-any.whl
    ```
  - Test your package in the virtual environment's Python interpreter.
  - Deactivate the virtual environment:
    ```
    deactivate
    ```

7. **Push Your Changes:**  
Once you've tested your changes locally, commit them and push to your fork on GitHub:

```
git add .
git commit -m "Description of your changes"
git push origin my-feature-branch
```


8. **Create a Pull Request:**  
Go to the main repository on GitHub and create a new pull request from your feature or bugfix branch.

9. **Address Review Feedback:**  
If there are any changes suggested during the review of your pull request, make the changes in your local environment, commit them, and push to your branch on GitHub. The pull request will update automatically.

Thank you for contributing to `rimo`!

## License
[License information]
