from setuptools import setup, find_packages

setup(
    name='robinBots',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pytest',
        'python-dotenv',
        # Add other dependencies as needed
    ],
    entry_points={
        'console_scripts': [
            'robinbots=src.main:main',
        ],
    },
)

