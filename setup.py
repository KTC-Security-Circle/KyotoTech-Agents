from setuptools import setup, find_packages
import sys
sys.path.append('./src')
# sys.path.append('./test')

def _from_file(file_name):
    return open(file_name).read().splitlines()


setup(
    name='tech_agents',
    version='0.0.6',
    author='Yuki Kimoto',
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=_from_file('requirements.txt'),
    license='LICENSE.txt',
)
