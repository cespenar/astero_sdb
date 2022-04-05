import os
import shutil
from pathlib import Path
from zipfile import ZipFile

import mesa_reader as mesa
import pandas as pd
from sqlalchemy import create_engine

from .gyre_reader import GyreData


class SdbGrid:
    """Structure containing a processed MESA grid of sdB stars.

    Reads a grid and provides methods to extract and read models.

    Parameters
    ----------
    db_file : str
        Database containing the grid of models.
    grid_dir : str
        Directory containing the zipped grid of models.

    Attributes
    ----------
    db_file : str
        Path to the input database. 
    grid_dir : str
        Path to the directory containing the zipped grid of models.
    data : pandas.DataFrame
        Pandas DataFrame containing the grid.

    Examples
    ----------
    >>> database = Path('sdb_grid_cpm.db')
    >>> grid_dir = Path('grid_sdb')
    >>> g = SdbGrid(database, grid_dir)

    Here `database` is the database containing the processed grid of calcualted
    MESA sdB models and `grid_dir` is the directory containing the full
    compressed grid. The grid is then initialized.
    """

    def __init__(self, db_file: Path, grid_dir: Path):
        """Creates SdbGrid object from a processed grid of MESA sdB models.

        Parameters
        ----------
        db_file : Path
            Database containing the grid of models.
        grid_dir : Path
            Directory containing the zipped grid of models.
        """

        self.db_file: Path = db_file
        self.grid_dir: Path = grid_dir
        engine = create_engine(f'sqlite:///{self.db_file}')
        self.data = pd.read_sql('models', engine)

    def __str__(self):
        return (f'SdbGrid based on "{self.db_file}" '
                f'database and with models located at "{self.grid_dir}"')

    def __repr__(self):
        return f'SdbGrid(db_file={self.db_file}, grid_dir={self.grid_dir})'

    def read_history(self,
                     log_dir: str,
                     top_dir: str,
                     dest_dir: Path = Path('.'),
                     delete_file: bool = True,
                     rename: bool = False,
                     keep_tree: bool = False) -> mesa.MesaData:
        """Reads a MESA history file and returns a MesaData object.

        Parameters
        ----------
        log_dir : str
            Log directory.
        top_dir : str
            Top directory.
        dest_dir : Path, optional
            Temporary directory for the required track. Default: Path('.').
        delete_file : bool, optional
            If True delete the extracted track. The track is not deleted if
            'keep_tree' is True. Default: True.
        rename : bool, optional
            If True it renames the history file to include information about
            the model contained in log_dir.
        keep_tree : bool, optional
            If True extract file with its directory structure (default
            ZipFile.extract() behaviour), otherwise extract file directly to
            'dest_dir'. Default: False.

        Returns
        ----------
        MesaData
            Evolutionary track (MESA history file) as MesaData object.
        """

        if rename:
            history_name = f'history{log_dir[4:]}.data'
        else:
            history_name = 'history.data'
        if keep_tree:
            file_name = dest_dir.joinpath(top_dir, log_dir, history_name)
        else:
            file_name = dest_dir.joinpath(history_name)
        if not self.model_extracted(file_name):
            self.extract_history(log_dir, top_dir, dest_dir, rename, keep_tree)
        data = mesa.MesaData(str(file_name))
        if delete_file and not keep_tree:
            file_name.unlink()
        return data

    def read_evol_model(self,
                        log_dir: str,
                        top_dir: str,
                        he4: float,
                        dest_dir: Path = Path('.'),
                        delete_file: bool = True,
                        keep_tree: bool = False) -> mesa.MesaData:
        """Reads a single evolutionary model (a profile) and returns a MesaData
        object.

        Parameters
        ----------
        log_dir : str
            Log directory.
        top_dir : str
            Top directory.
        he4 : float
            Central helium abundance of the required model.
        dest_dir : Path, optional
            Directory for the required model. Default: Path('.').
        delete_file : bool, optional
            If True delete the extracted model. The model is not deleted if
            'keep_tree' is True. Default: True.
        keep_tree : bool, optional
            If True extract file with its directory structure (default
            ZipFile.extract() behaviour), otherwise extract file directly to
            'dest_dir'. Default: False.

        Returns
        ----------
        MesaData
            Evolutionary model (MESA profile file) as MesaData object.
        """

        if keep_tree:
            file_name = dest_dir.joinpath(top_dir, log_dir,
                                          self.evol_model_name(he4))
        else:
            file_name = dest_dir.joinpath(self.evol_model_name(he4))
        if not self.model_extracted(file_name):
            self.extract_evol_model(log_dir, top_dir, he4, dest_dir, keep_tree)
        data = mesa.MesaData(str(file_name))
        if delete_file and not keep_tree:
            file_name.unlink()
        return data

    def read_puls_model(self,
                        log_dir: str,
                        top_dir: str,
                        he4: float,
                        dest_dir: Path = Path('.'),
                        delete_file: bool = True,
                        keep_tree=False) -> GyreData:
        """Reads a calculated GYRE model and returns a GyreData object.

        Parameters
        ----------
        log_dir : str
            Log directory.
        top_dir : str
            Top directory.
        he4 : float
            Central helium abundance of the required model.
        dest_dir : Path, optional
            Temporary directory for the required model. Default: Path('.').
        delete_file : bool, optional
            If True delete the extracted model. The model is not deleted if
            'keep_tree' is True. Default: True.
        keep_tree : bool, optional
            If True extract file with its directory structure (default
            ZipFile.extract() behaviour), otherwise extract file directly to
            'dest_dir'. Default: False.

        Returns
        ----------
        GyreData
            Pulsation model as GyreData object.
        """

        if keep_tree:
            file_name = dest_dir.joinpath(top_dir, log_dir,
                                          self.puls_model_name(he4))
        else:
            file_name = dest_dir.joinpath(self.puls_model_name(he4))
        if not self.model_extracted(file_name):
            self.extract_puls_model(log_dir, top_dir, he4, dest_dir, keep_tree)
        data = GyreData(file_name)
        if delete_file and not keep_tree:
            file_name.unlink()
        return data

    def extract_history(self,
                        log_dir: str,
                        top_dir: str,
                        dest_dir: Path,
                        rename: bool = False,
                        keep_tree: bool = False) -> None:
        """Extracts a MESA history file.

        Parameters
        ----------
        log_dir : str
            Log directory.
        top_dir : str
            Top directory.
        dest_dir : Path
            Destination directory for the extracted model if 'keep_tree' is
            False, or the root directory if 'keep_tree' is True.
        rename : bool
            If True it renames the history file to include information about
            the model contained in log_dir. Default: False.
        keep_tree : bool, optional
            If True extract file with its directory structure (default
            ZipFile.extract() behaviour), otherwise extract file directly to
            'dest_dir'. Default: False.

        Returns
        ----------
        """

        grid_zip_file = self.grid_dir.joinpath(self.archive_name(top_dir))
        if rename:
            history_name = f'history{log_dir[4:]}.data'
        else:
            history_name = 'history.data'
        grid_zip_path = os.path.join(top_dir, log_dir, history_name)
        dest_path = dest_dir.joinpath(history_name)

        with ZipFile(grid_zip_file) as archive:
            if keep_tree:
                archive.extract(grid_zip_path, dest_dir)
            else:
                with archive.open(grid_zip_path) as zipped_file, open(
                        dest_path, 'wb') as dest_file:
                    shutil.copyfileobj(zipped_file, dest_file)

    def extract_evol_model(self,
                           log_dir: str,
                           top_dir: str,
                           he4: float,
                           dest_dir: Path,
                           keep_tree: bool = False) -> None:
        """Extracts a single evolutionary model (a profile).

        Parameters
        ----------
        log_dir : str
            Log directory.
        top_dir : str
            Top directory.
        he4 : float
            Central helium abundance of the required model.
        dest_dir : Path
            Destination directory for the extracted model if 'keep_tree' is
            False, or the root directory if 'keep_tree' is True.
        keep_tree : bool, optional
            If True extract file with its directory structure (default
            ZipFile.extract() behaviour), otherwise extract file directly to
            'dest_dir'. Default: False.

        Returns
        ----------
        """

        grid_zip_file = self.grid_dir.joinpath(self.archive_name(top_dir))
        model_name = self.evol_model_name(he4)
        grid_zip_path = os.path.join(top_dir, log_dir, model_name)
        dest_path = dest_dir.joinpath(model_name)

        with ZipFile(grid_zip_file) as archive:
            if str(grid_zip_path) in archive.namelist():
                if keep_tree:
                    archive.extract(grid_zip_path, dest_dir)
                else:
                    with archive.open(grid_zip_path) as zipped_file, open(
                            dest_path, 'wb') as dest_file:
                        shutil.copyfileobj(zipped_file, dest_file)

    def extract_puls_model(self,
                           log_dir: str,
                           top_dir: str,
                           he4: float,
                           dest_dir: Path,
                           keep_tree: bool = False) -> None:
        """Extracts a single calculated GYRE model.

        Parameters
        ----------
        log_dir : str
            Log directory.
        top_dir : str
            Top directory.
        he4 : float
            Central helium abundance of the required model.
        dest_dir : Path
            Destination directory for the extracted model if 'keep_tree' is
            False, or the root directory if 'keep_tree' is True.
        keep_tree : bool, optional
            If True extract file with its directory structure (default
            ZipFile.extract() behaviour), otherwise extract file directly to
            'dest_dir'. Default: False.

        Returns
        ----------
        """

        grid_zip_file = self.grid_dir.joinpath(self.archive_name(top_dir))
        model_name = self.puls_model_name(he4)
        grid_zip_path = os.path.join(top_dir, log_dir, model_name)
        dest_path = dest_dir.joinpath(model_name)

        with ZipFile(grid_zip_file) as archive:
            if str(grid_zip_path) in archive.namelist():
                if keep_tree:
                    archive.extract(grid_zip_path, dest_dir)
                else:
                    with archive.open(grid_zip_path) as zipped_file, open(
                            dest_path, 'wb') as dest_file:
                        shutil.copyfileobj(zipped_file, dest_file)

    def extract_gyre_input_model(self,
                                 log_dir: Path,
                                 top_dir: Path,
                                 he4: float,
                                 dest_dir: Path,
                                 keep_tree: bool = False) -> None:
        """Extracts a single GYRE input model.

        Parameters
        ----------
        log_dir : Path
            Log directory.
        top_dir : Path
            Top directory.
        he4 : float
            Central helium abundance of the required model.
        dest_dir : Path
            Destination directory for the extracted file if 'keep_tree' is
            False, or the root directory if 'keep_tree' is True.
        keep_tree : bool, optional
            If True extract file with its directory structure (default
            ZipFile.extract() behaviour), otherwise extract file directly to
            'dest_dir'. Default: False.

        Returns
        ----------
        """

        grid_zip_file = self.grid_dir.joinpath(self.archive_name(top_dir))
        model_name = self.gyre_input_name(he4)
        grid_zip_path = top_dir.joinpath(log_dir, model_name)
        dest_path = dest_dir.joinpath(model_name)

        with ZipFile(grid_zip_file) as archive:
            if str(grid_zip_path) in archive.namelist():
                if keep_tree:
                    archive.extract(grid_zip_path, dest_dir)
                else:
                    with archive.open(grid_zip_path) as zipped_file, open(
                            dest_path, 'wb') as dest_file:
                        shutil.copyfileobj(zipped_file, dest_file)

    def extract_log_dir(self,
                        log_dir: Path,
                        top_dir: Path,
                        dest_dir: Path):
        """Extracts a MESA log directory.

        Parameters
        ----------
        log_dir : Path
            Log directory.
        top_dir : Path
            Top directory.
        dest_dir : Path
            Destination directory for the extracted directory tree.

        Returns
        ----------
        """

        grid_zip_file = self.grid_dir.joinpath(self.archive_name(top_dir))
        grid_zip_path = top_dir.joinpath(log_dir)

        with ZipFile(grid_zip_file) as archive:
            for f_name in archive.namelist():
                if f_name.startswith(str(grid_zip_path)):
                    archive.extract(f_name, dest_dir)

    def evol_model_exists(self,
                          log_dir: Path,
                          top_dir: Path,
                          he4: float) -> bool:
        """Checks if a profile exists in archive.

        Parameters
        ----------
        log_dir : Path
            Log directory.
        top_dir : Path
            Top directory.
        he4 : float
            Central helium abundance of the required model.

        Returns
        ----------
        bool
            True if a profile exists, False otherwise.
        """

        grid_zip_file = self.grid_dir.joinpath(self.archive_name(top_dir))
        model_name = self.evol_model_name(he4)
        grid_zip_path = top_dir.joinpath(log_dir, model_name)

        with ZipFile(grid_zip_file) as archive:
            if str(grid_zip_path) in archive.namelist():
                return True
            else:
                return False

    def puls_model_exists(self,
                          log_dir: Path,
                          top_dir: Path,
                          he4: float) -> bool:
        """Checks if a calculated GYRE model exists in archive.

        Parameters
        ----------
        log_dir : Path
            Log directory.
        top_dir : Path
            Top directory.
        he4 : float
            Central helium abundance of the required model.

        Returns
        ----------
        bool
            True if a profile exists, False otherwise.
        """

        grid_zip_file = self.grid_dir.joinpath(self.archive_name(top_dir))
        model_name = self.puls_model_name(he4)
        grid_zip_path = top_dir.joinpath(log_dir, model_name)

        with ZipFile(grid_zip_file) as archive:
            if str(grid_zip_path) in archive.namelist():
                return True
            else:
                return False

    def gyre_input_exists(self,
                          log_dir: Path,
                          top_dir: Path,
                          he4: float) -> bool:
        """Checks if a GYRE input model exists in archive.

        Parameters
        ----------
        log_dir : Path
            Log directory.
        top_dir : Path
            Top directory.
        he4 : float
            Central helium abundance of the required model.

        Returns
        ----------
        bool
            True if a profile exists, False otherwise.
        """

        grid_zip_file = self.grid_dir.joinpath(self.archive_name(top_dir))
        model_name = self.gyre_input_name(he4)
        grid_zip_path = top_dir.joinpath(log_dir, model_name)

        with ZipFile(grid_zip_file) as archive:
            if str(grid_zip_path) in archive.namelist():
                return True
            else:
                return False

    @staticmethod
    def model_extracted(path: Path) -> bool:
        """Checks if model is already extracted.

        Parameters
        ----------
        path : Path
            Path to the model.

        Returns
        ----------
        bool
            True if model exists, False otherwise.
        """

        if path.is_file():
            return True
        else:
            return False

    @staticmethod
    def archive_name(top_dir: str) -> str:
        """Returns a name of a zip file containing a top directory.

        Parameters
        ----------
        top_dir : str
            Top directory.

        Returns
        ----------
        str
            Name of zipfile.
        """

        return f'grid{top_dir[4:]}.zip'

    @staticmethod
    def evol_model_name(he4: float) -> str:
        """Returns a name of a MESA profile for helium abundance 'he4'.

        Parameters
        ----------
        he4 : float
            Central helium abundance of the model.

        Returns
        ----------
        str
            Name of MESA profile.
        """

        return f'custom_He{round(he4, 6)}.data'

    @staticmethod
    def puls_model_name(he4: float) -> str:
        """Returns a name of a calculated GYRE model for helium abundance
        'he4'.

        Parameters
        ----------
        he4 : float
            Central helium abundance of the model.

        Returns
        ----------
        str
            Name of calculated GYRE model.
        """

        return f'custom_He{round(he4, 6)}_summary.txt'

    @staticmethod
    def gyre_input_name(he4: float) -> str:
        """Returns a name of an input model for GYRE model for helium
        abundance 'he4'.

        Parameters
        ----------
        he4 : float
            Central helium abundance of the model.

        Returns
        ----------
        str
            Name of GYRE input model.
        """

        return f'custom_He{round(he4, 6)}.data.GYRE'
