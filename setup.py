from setuptools import setup

setup(name='astero_sdb',
      version='0.0.33',
      description='Tools for asteroseismology of sdB stars using MESA and GYRE models.',
      url='https://github.com/cespenar/astero_sdb',
      author='Jakub Ostrowski',
      author_email='cespenar1@gmail.com',
      license='MIT',
      packages=['astero_sdb'],
      install_requires=[
          'mesa_reader',
          'sqlalchemy',
          'tqdm',
      ])
