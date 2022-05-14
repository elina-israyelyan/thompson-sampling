from setuptools import setup, find_packages

# Adding install requirements
setup(
    author="Elina Israyelyan",
    description="A package for implementing Thompson Sampling algorithm.",
    name="thompson_sampling",
    packages=find_packages(include=["thompson_sampling", "thompson_sampling.*"]),
    version="0.0.1",
    install_requires=['numpy>=1.22.3', 'pandas>=1.4.2', "plotly>=5.7.0", "scipy>=1.8.0"],
    python_requires=">=3.8, ~=3.10",
)
