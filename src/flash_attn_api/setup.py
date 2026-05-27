from setuptools import setup, find_packages
import os
from pathlib import Path
import subprocess


setup(
    name="flash_attn_turing_api",
    version='0.0.1',
    packages=find_packages(),
    install_requires=['flash_attn_turing'],
)

