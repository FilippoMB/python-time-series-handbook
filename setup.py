from setuptools import setup, find_packages

setup(
    name="tsa_course",
    version="0.3.1",
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'numpy>1.19.5',
        'matplotlib',
        'scipy',
        'tqdm'
    ],
    author="Filippo Maria Bianchi",
    author_email="filippombianchi@gmail.com",
    description="A collection of scripts and functions used in the course 'Time Series Analysis with Python'",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    project_urls={
        "Documentation": "https://filippomb.github.io/python-time-series-handbook",
        "Source Code": "https://github.com/FilippoMB/python-time-series-handbook",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)