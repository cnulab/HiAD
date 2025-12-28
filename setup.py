import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="hiad",
    version="0.2.0",
    description="high-resolution anomaly detection",
    author="cnulab",
    url="https://github.com/cnulab/HiAD",
    author_email="2024010482@bupt.cn",
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    python_requires='>=3.8',
    include_package_data=True,
    extras_require={
        'cuda': ['faiss-gpu'],
        'cuda11': ['faiss-gpu-cu11'],
        'cuda12': ['faiss-gpu-cu12'],
    }


)

