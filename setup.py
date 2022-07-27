from setuptools import setup

setup(name='astero_sdb',
      version='0.1.6',
      description='Tools for asteroseismology of sdB stars using MESA and GYRE models.',
      url='https://github.com/cespenar/astero_sdb',
      author='Jakub Ostrowski',
      author_email='cespenar1@gmail.com',
      license='MIT',
      packages=['astero_sdb'],
      install_requires=[
          'matplotlib',
          'mesa_reader',
          'numpy',
          'pandas',
          'sqlalchemy',
          'tqdm',
      ])
