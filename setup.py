# -*- coding: utf-8 -*- %reset -f
"""
@author: Hiromasa Kaneko
"""
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE.txt') as f:
    license = f.read()

setup(
    name='dcekit',
    version='2.13.1',
    description='Data Chemical Engineering toolkit',
    long_description=readme,
    author='Hiromasa Kaneko',
    author_email='hkaneko226@gmail.com',
    url='https://github.com/hkaneko1985/dcekit/',
    license=license,
    install_requires=['numpy', 'pandas', 'scikit-learn', 'scipy'],
    packages=find_packages()
)