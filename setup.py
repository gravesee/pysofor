from setuptools import setup

setup(
    name='pysofor',
    version='0.1',
    description='The funniest joke in the world',
    url='http://github.com/storborg/funniest',
    author='Eric Graves',
    author_email='flyingcircus@example.com',
    license='MIT',
    install_requires=[
        'numpy',
    ],
    packages=['pysofor'],
    zip_safe=False)