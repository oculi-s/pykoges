from setuptools import setup, find_packages

setup(
    name="pykoges",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[],
    entry_points={},
    author="oculis",
    author_email="oculis0925@yonsei.ac.kr",
    description="Python module for cohort data reading",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/oculi-s/pykoges",
)
