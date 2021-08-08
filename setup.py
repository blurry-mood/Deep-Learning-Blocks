from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Read the requirements from the TXT file
with open(path.join(here, 'requirements.txt')) as f:
    requirements = [req for req in f.read().split('\n') if not ('#' in req or req == '')]
        
# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

n = len('https://github.com/blurry-mood/Deep-Learning-Blocks/blob/main/')
j = 0
# Make readme consistent in pypi server
for _ in range(len(long_description)-6):
    if long_description[j:j+5] == 'docs/':
        long_description = long_description[:j] + 'https://github.com/blurry-mood/Deep-Learning-Blocks/blob/main/' + long_description[j:]
        j += n
        print(j)
    j+=1

print(long_description)

exec(open('deepblocks/version.py').read())

setup(
    name='deepblocks',
    version=__version__,
    author='Ayoub Assis',
    author_email='assis.ayoub@gmail.com',
    url='https://github.com/blurry-mood/Deep-Learning-Blocks',
    license='LICENSE',

    packages=find_packages(exclude=['tests','notebooks']),

    keywords='pytorch cnn layers',
    description='Useful PyTorch Layers',
    long_description=long_description,
    long_description_content_type='text/markdown',

    install_requires=requirements,
    python_requires='>=3.7',

)
