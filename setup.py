from setuptools import setup

setup(
    name='src',
    packages=[''],
    version='0.0.2',
    description='',
    author='Valentino Constantinou',
    author_email='vconstan@jpl.nasa.gov',
    url='https://github.com/vc1492a/sTEC-d-dt-Anomaly-Detection',
    download_url='https://github.com/vc1492a/sTEC-d-dt-Anomaly-Detection/archive/0.0.2.tar.gz',
    keywords=['tsunami', 'anomaly', 'detection'],
    classifiers=[],
    license='Apache License, Version 2.0',
    install_requires=[
        'pandas',
        'scikit-learn',
        'tqdm'
    ]
)
