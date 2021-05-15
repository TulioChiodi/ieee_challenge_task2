from setuptools import find_packages, setup

setup(
    name='src',
    version='0.1.0',
    packages=find_packages(),   
    install_requires=[
        'jiwer==2.2.0',
        'librosa==0.8.0',
        'numpy==1.18.1',
        'pandas==1.0.3',
        'pystoi==0.3.3',
        'scipy==1.4.1',
        'soundfile==0.10.3.post1',
        'torch==1.0.1',
        'transformers==4.4.2',
        'tqdm==4.36.1',
        'wget==3.2',
    ]
)   
