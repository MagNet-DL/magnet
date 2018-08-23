#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='MagNet',
    version='0.1',
    description='MagNet makes it stupid simple to create Deep Learning projects',
    author='Vaisakh',
    author_email='svaisakh1994@gmail.com',
    url='https://github.com/svaisakh/magnet',
    packages=['magnet'],
    license='MIT license',
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=[
        'torch',
        'torchvision'
        ]
)