from setuptools import setup

setup(
    name='seghub',
    version='0.0.1',
    description='Tools for segmentation, with focus on combining variable feature extractors with a random forest classifier.',
    author='Roman Schwob',
    author_email='roman.schwob@students.unibe.ch',
    license='GNU GPLv3',
    packages=['seghub'],
    package_dir={'': 'src'},
    zip_safe=False
    )