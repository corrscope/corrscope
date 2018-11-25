from setuptools import setup

setup(
    name='ovgenpy',
    version='0',
    packages=[''],
    url='',
    license='BSD-2-Clause',
    author='jimbo1qaz',
    author_email='',
    description='',
    tests_require=['pytest', 'pytest-pycharm', 'hypothesis', 'delayed-assert'],
    install_requires=['numpy', 'scipy', 'click', 'matplotlib', 'ruamel.yaml',
                      'attrs>=18.2.0']
)
