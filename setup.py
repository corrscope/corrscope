from setuptools import setup

setup(
    name='ovgenpy',
    version='0',
    packages=['ovgenpy'],
    url='https://github.com/jimbo1qaz/ovgenpy',
    license='BSD-2-Clause',
    author='jimbo1qaz',
    author_email='',
    description='',
    # https://docs.pytest.org/en/latest/goodpractices.html
    setup_requires=["pytest-runner"],
    tests_require=['pytest>=3.2.0', 'pytest-pycharm', 'hypothesis', 'delayed-assert'],
    install_requires=[
        'ruamel.yaml>=0.15.70',  # See test_config.py to pick a suitable minimum version
        'numpy', 'scipy', 'click', 'more_itertools',
        'matplotlib',
        'attrs>=18.2.0',
        'PyQt5',
    ]
)
