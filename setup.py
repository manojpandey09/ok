from setuptools import find_packages, setup

setup(
    name='DiamondPricePrediction',
    version='0.0.1',
    author='Manoj Pandey',
    author_email='pandeymannu54@gmail.com',
    install_requires=[
        "scikit-learn",
        "pandas",
        "numpy"
    ],
    packages=find_packages()
)
