from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('MIT-LICENSE.txt') as f:
    license = f.read()

setup(
    name='SparseAutoencoder',
    version='0.0.1',
    description='Package of the sparse autoencoder',
    long_description=readme,
    author='Ryo Ozaki',
    author_email='ryo.ozaki@em.ci.ritsumei.ac.jp',
    url='https://github.com/RyoOzaki/SparseAutoencoder',
    license=license,
    install_requires=['numpy', 'tensorflow'],
    packages=find_packages(exclude=('SparseAutoencoder','SparseAutoencoder.util'))
)
