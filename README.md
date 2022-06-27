# astero_sdb

***
_astero_sdb_ is the library containing the set of tools for
asteroseismology of sdB stars using the
[grid](https://sdb-grid-viewer.herokuapp.com) of evolutionary
MESA models and pulsation GYRE models of sdB stars calculated for
the [ARDASTELLA](https://ardastella.up.krakow.pl/) research group.

## Content

***
The package consists of five modules:

* #### `sdb_grid_reader.py`
* #### `star.py`
* #### `gyre_reader.py`
* #### `utils.py`
* #### `plots.py`

Detailed docstrings are provided for all methods and functions, except for
functions in `plots.py` (TBD).

## Installation

***
The package can be installed using pip and GitHub repository:

    pip install git+https://github.com/cespenar/astero_sdb.git

It will be added to PyPI with the next large release.

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
as an `SdbGrid` object, and `conditions` are sample constraints put on the
grid.

## Acknowledgements

***
The author was financially supported by the Polish National Science Centre
grant UMO-2017/26/E/ST9/00703. The grid of models was calculated using the
resources provided by
[Wrocław Centre for Networking and Supercomputing](https://www.wcss.pl/en/),
grant No. 265.
