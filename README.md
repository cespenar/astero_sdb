# Astero-sdB

***
_Astero-sdB_ is the library containing the set of tools for
asteroseismology of sdB stars using the
[grid](https://sdb-grid-viewer.herokuapp.com) of evolutionary
MESA models and pulsation GYRE models of sdB stars calculated for
the [ARDASTELLA](https://ardastella.up.krakow.pl/) research group.

## Content

***
The package consists of five modules:

1. #### `sdb_grid_reader.py`
The module allows to read the processed grid and store it as the `SdbGrid`
class. The structure provides methods to extract and read evolutionary
and pulsation models.

2. #### `star.py`
The module contains the `Star` class, which includes observational properties
of a target star. The class contains methods for fitting a star to the grid.

3. #### `gyre_reader.py`
The tool for reading output of the GYRE pulsation code and store it in the
`GyreData` class. The module is useful for any GYRE models and its application
is not limited to the sdB grid.

4. #### `utils.py`
Various stand-alone utility functions.

5. #### `plots.py`
The most common plots useful during fitting target stars
to the grid of models. In general, not recommended for publication, but
excellent for quick analysis of the results.

## Installation

***
First, install the _mesa_reader_, which is not available in PyPi:

    pip install git+https://github.com/wmwolf/py_mesa_reader.git

Then, _Astero-sdB_ can be installed using pip:

    pip install git+https://github.com/cespenar/astero_sdb.git

Unfortunately, the package cannot be currently uploaded to PyPI, because one
of its dependencies, _mesa-reader_, is not yet been available on PyPi.
Installing from GitHub using _pip_ is not supported during uploading a
package to PyPi or during the installation of requirements, hence the more
cumbersome installation is currently required.

_Astero-sdB_ was tested with Python 3.9 and 3.10.

## Basic usage

***

### Read the grid of models:

```python
from pathlib import Path
from astero_sdb.sdb_grid_reader import SdbGrid

database = Path('/Users/cespenar/sdb/sdb_grid_cpm.db')
grid_dir = Path('/Volumes/T3_2TB/sdb/grid_sdb')
g = SdbGrid(database, grid_dir)
```

where `database` is a SQLite database containing the processed grid,
`grid_dir` the directory containing compressed models, and `g` is the read grid
as an `SdbGrid` object.

### Finding the best models

A basic example how to find the best fit for a target star on a constrained
grid:

```python
import pandas as pd
from pathlib import Path
from astero_sdb.sdb_grid_reader import SdbGrid
from astero_sdb.star import Star

pd.options.mode.chained_assignment = None

target = Star(name='test_star',
              t_eff=25790.0, t_eff_err_p=160.0, t_eff_err_m=160.0,
              log_g=5.43, log_g_err_p=0.01, log_g_err_m=0.01,
              frequencies_list='test_star_frequencies.txt')

database = Path('/Users/cespenar/sdb/sdb_grid_cpm.db')
grid_dir = Path('/Volumes/T3_2TB/sdb/grid_sdb')
g = SdbGrid(database, grid_dir)

conditions = \
    (g.data.m_i <= 2.0) & \
    (g.data.m_env <= 0.01) & \
    (g.data.z_i <= 0.035)

df = g.data[conditions]

target.evaluate_chi2(df_selected=df,
                     grid=g,
                     dest_dir=grid_dir,
                     use_spectroscopy=True,
                     save_period_list=True,
                     period_list_name=f'{target.name}_periods.txt',
                     results_file_name=f'{target.name}_results.txt')
```

where `target` contains the properties of the target star as a Star object,
`database` is a SQLite database containing the processed grid,
`grid_dir` the directory containing compressed models, `g` is the read grid
as an `SdbGrid` object, and `conditions` are sample constraints imposed on the
grid.

## Acknowledgements

***
The author was financially supported by the Polish National Science Centre
grant UMO-2017/26/E/ST9/00703. The grid of models was calculated using the
resources provided by
[Wrocław Centre for Networking and Supercomputing](https://www.wcss.pl/en/),
grant No. 265.
