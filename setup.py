from setuptools import setup, find_packages

setup(
    name="heart-disease-classifier",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "pillow>=10.0.0",
        "transformers>=4.35.0",
        "scikit-learn>=1.3.0",
        "openpyxl>=3.1.0",
        "python-dotenv>=1.0.0",
        "torch>=2.0",
        "torchvision>=0.15",
    ],
    python_requires=">=3.9",
)
