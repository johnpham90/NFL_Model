from setuptools import setup, find_packages

setup(
    name="nfl_model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'psycopg2-binary',
        'python-dotenv',
        'sqlalchemy',
    ],
    author="John Pham",
    description="NFL Prediction Model",
    python_requires='>=3.8',
)