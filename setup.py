from setuptools import setup

setup(
    name='ovgenpy',
    version='0',
    packages=[''],
    url='',
    license='BSD-2-Clause',
    author='nyanpasu64',
    author_email='',
    description='',
    tests_require=['pytest>=3.2.0', 'pytest-pycharm', 'hypothesis', 'delayed-assert'],
    install_requires=[
        # ruamel.yaml 0.15.55 implements slice assignment in loaded list (CommentedSeq).
        'numpy', 'scipy', 'click', 'ruamel.yaml>=0.15.55', 'more_itertools',
        'matplotlib',
        'attrs>=18.2.0',
        'PyQt5',
    ]
)
