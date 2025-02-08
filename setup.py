from setuptools import setup, find_packages

setup(
    name='blurred-thoughts-SFT',
    version='0.1.0',
    packages=find_packages(include=['btsft', 'btsft.*']),
    package_data={
        'btsft': ['config/*.yaml'],
    },
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'datasets>=2.12.0',
        'bitsandbytes>=0.39.0',
        'unsloth>=0.1.0',
        'pyyaml>=6.0',        
        'typing-extensions',   
        'tqdm',               
        'wandb',              
        'numpy>=1.24.0',      
        'accelerate>=0.20.0', 
        'tensorboard',        
    ],
    entry_points={
        'console_scripts': [
            'btsft=btsft.main:main',
        ],
    },
    python_requires='>=3.8',
    author='Your Name',
    description='A package for training language models with Blurred Thoughts SFT',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)