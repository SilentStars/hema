from setuptools import setup, find_packages

VERSION = '0.1.3'
DESCRIPTION = 'EMG analysis toolbox by Hou.'

setup(
    name='hema',
    version=VERSION,
    author='Qinhan Hou',
    author_email='qinhan.hou@idsia.ch',
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    packages=find_packages(),
    install_requires=['torch', 'torchvision', 'torchaudio', 'numpy', 'scipy'],
    keyword=['emg'],
    classifiers=[]
)