from setuptools import setup

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setup(name='dimenet',
      version='1.0',
      description='Directional Message Passing for Molecular Graphs',
      author='Johannes Klicpera, Janek Groß, Stephan Günnemann',
      author_email='klicpera@in.tum.de',
      packages=['dimenet'],
      install_requires=install_requires,
      zip_safe=False)
