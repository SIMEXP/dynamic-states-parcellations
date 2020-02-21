from setuptools import setup

setup(
    name='dynamicparcels',
    version='0.1.1',
    license='MIT',
    description='This python script generates dynamic states of parcellation of functional MRI data at the individual level.',

    author='Amal Boukhdhir',
    author_email='boukhdhiramal@yahoo.fr',
    url='',
    packages=['dynpar'],

    install_requires=[ 'seaborn','pytest', 'numpy', 'nilearn', 'nibabel', 'scipy',
                      'sklearn', 'matplotlib', 'nose', "dask[array,dataframe]>=1.0.0",
                      'psutil', 'bokeh', 'dask', 'joblib', 'scikit-image','pathos',
                      'dash', 'dash_core_components', 'dash_html_components', 'SharedArray',
                      'pandas_datareader', 'toolz', 'cytoolz', 'cachey', 'pyclustering', 'h5py', 'cufflinks'],
)
