from setuptools import setup

install_requires = [
        "numpy",
        "scipy>=1.3",
        "sympy>=1.5",
        "tensorflow>=2.1",
        "tensorflow_addons",
        "tqdm",
]

setup(
        name='dimenet',
        version='1.0',
        description='Directional Message Passing for Molecular Graphs',
        author='Johannes Klicpera, Janek Groß, Stephan Günnemann',
        author_email='klicpera@in.tum.de',
        packages=['dimenet'],
        install_requires=install_requires,
        zip_safe=False,
)
