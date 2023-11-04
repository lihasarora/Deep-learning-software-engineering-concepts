# this is responsible for setting up our python code as a package.
# Anyone can install it if we deploy it to Python PYPI

# This is coupled with the __init__.py file which tells find_packages() function to execute everything written here

from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This function returns the list of requirements
    '''
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [x.replace("\n","") for x in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    
    name = 'myproject',
    version = '0.0.1',
    author = 'sahil',
    author_email= 'sahil.arora.rmd@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt') 
)