from setuptools import setup, find_packages


setup(
    name='bgtrees',
    version='0.0.1',
    authors=[
        'Giuseppe De Laurentis',
        'Mathieu Pellen',
        'Juan Cruz-Martinez'
    ],
    description='',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
    ],
)
