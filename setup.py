from setuptools import setup, find_packages

setup(
    name="mmidas",
    version="0.1.1",
    author="Yeganeh Marghi",
    author_email="yeganeh.marghi@alleninstitute.org",
    description="Couple Mixture VAE models for single cell data",
    packages=find_packages(),
    install_requires=[],
    keywords=['VAE', 'latent representation', 'single-cell', 'unimodal', 'multiodal', 'mixture model', 'coupled model'],
    classifiers=["Programming Language :: Python :: 3"],
    python_requires='>=3.9'
)
