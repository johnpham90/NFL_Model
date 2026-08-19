from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "NFL Prediction Model for analyzing and predicting NFL game outcomes"

setup(
    name="nfl_model",
    version="0.1.0",
    author="John Pham",
    author_email="john.pham@email.com",  # ADD YOUR EMAIL
    description="NFL Prediction Model for analyzing and predicting game outcomes",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/johnpham90/NFL_Model",
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # COMPLETE DEPENDENCIES
    install_requires=[
        # Data Processing
        'pandas>=1.5.0,<3.0',
        'numpy>=1.21.0,<2.0',
        
        # Machine Learning
        'scikit-learn>=1.2.0,<2.0',
        'xgboost>=1.7.0,<3.0',
        'lightgbm>=3.3.0,<5.0',
        'optuna>=3.0.0,<5.0',
        
        # Database
        'psycopg2-binary>=2.9.0,<3.0',
        'sqlalchemy>=1.4.0,<3.0',
        
        # Configuration & Utils
        'python-dotenv>=1.0.0,<2.0',
        'pyyaml>=6.0,<7.0',
        'tqdm>=4.64.0,<5.0',
        
        # Data Visualization (optional core)
        'matplotlib>=3.6.0,<4.0',
        'seaborn>=0.12.0,<1.0',
    ],
    
    # OPTIONAL DEPENDENCIES
    extras_require={
        'dev': [
            'jupyter>=1.0.0',
            'pytest>=7.2.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ],
        'api': [
            'fastapi>=0.95.0',
            'uvicorn>=0.20.0',
            'pydantic>=1.10.0',
        ],
        'visualization': [
            'plotly>=5.13.0',
            'dash>=2.8.0',
            'streamlit>=1.20.0',
        ],
        'all': [
            'jupyter>=1.0.0', 'pytest>=7.2.0', 'black>=23.0.0',
            'fastapi>=0.95.0', 'uvicorn>=0.20.0', 'pydantic>=1.10.0',
            'plotly>=5.13.0', 'dash>=2.8.0', 'streamlit>=1.20.0',
        ]
    },
    
    # CLI COMMANDS
    entry_points={
        'console_scripts': [
            'nfl-train=src.scripts.train_model:main',
            'nfl-predict=src.scripts.generate_predictions:main',
            'nfl-evaluate=src.scripts.evaluate_model:main',
        ],
    },
    
    # PACKAGE METADATA
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    python_requires='>=3.8,<4.0',
    
    # INCLUDE PACKAGE DATA
    include_package_data=True,
    package_data={
        'nfl_model': [
            'config/*.yaml',
            'config/*.yml',
            'data/external/*.csv',
        ],
    },
    
    # PROJECT URLS
    project_urls={
        "Bug Reports": "https://github.com/johnpham90/NFL_Model/issues",
        "Source": "https://github.com/johnpham90/NFL_Model",
        "Documentation": "https://github.com/johnpham90/NFL_Model/blob/main/README.md",
    },
    
    keywords="nfl, machine-learning, sports-analytics, prediction, football",
    license="MIT",
)