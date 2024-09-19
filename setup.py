from setuptools import setup, find_packages

setup(
    name="tml",  # The name of your package
    version="0.1",  # Initial version
    author="Ehsan Karimiara",
    author_email="e.karimiara@gmail.com",
    description="A package for transductive machine learning models",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ehsanka/TML",  # Your GitHub repo URL
    packages=find_packages(),  # Automatically find all packages in your project
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'torch',
        'pytorch-lightning',
        # Add other dependencies if necessary
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',  # Minimum Python version
)