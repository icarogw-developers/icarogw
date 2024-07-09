import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='icarogw',
    version='2.0.3',
    author='Simone Mastrogiovanni',
    author_email='simone.mastrogiovanni@ligo.org',
    description='A python package for inference of population properties of noisy, heterogeneous and incomplete observations',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/simone-mastrogiovanni/icarogw',
    license='EUPL-1.2',
    python_requires='>=3.10',
    packages=['icarogw'],
    install_requires=['bilby==2.3.0','ChainConsumer==1.1.1','mhealpy==0.3.3',
                     'ligo.skymap==2.0.1','mpmath==1.3.0']
)
