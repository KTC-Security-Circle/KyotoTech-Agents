from setuptools import setup, find_packages
import sys
sys.path.append('./src')
# sys.path.append('./test')

def requirements_from_file(file_name):
    return open(file_name).read().splitlines()

setup(
    name='tech_agents',
    version='0.0.2',
    author='Yuki Kimoto',
    packages=find_packages(),
    install_requires=requirements_from_file('requirements.txt'),
    license="LICENSE.txt",
)