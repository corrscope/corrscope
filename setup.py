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
    tests_require=['pytest', 'pytest-pycharm', 'hypothesis'],
    install_requires=['numpy', 'scipy', 'imageio', 'click', 'matplotlib',
                      'dataclasses;python_version<"3.7"', 'ruamel.yaml']
)
