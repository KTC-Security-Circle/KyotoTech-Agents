from setuptools import setup, find_packages

def _from_file(file_name):
    return open(file_name).read().splitlines()

setup(
    name='tech_agents',
    version='0.1.0',
    author='',
    description='tech_agents : agents use toolagent',
    packages=find_packages(),
    
    install_requires=_from_file('requirements.txt'),
    license='MIT',
    include_package_data=True,
)
