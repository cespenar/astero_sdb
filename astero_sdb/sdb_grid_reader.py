import os
import shutil
from zipfile import ZipFile

import mesa_reader as mesa
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sqlalchemy import create_engine

from .gyre_reader import GyreData
from .star import Star


class SdbGrid():
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
    >>> database = 'sdb_grid_cpm.db'
    >>> grid_dir = 'grid_sdb'
    >>> g = SdbGrid(database, grid_dir)

    Here `database` is the database containing the processed grid of
    calcualted MESA sdB models and `grid_dir` is the directory containing
    the full compressed grid. The grid is then initialized.
    """

    def __init__(self, db_file: str, grid_dir: str):
        """Creates SdbGrid object from a processed
        grid of MESA sdB models.

        Parameters
        ----------
        db_file : str
            Database containing the grid of models.
        grid_dir : str
            Directory containing the zipped grid of models.
        """

        self.db_file = db_file
        self.grid_dir = grid_dir
        engine = create_engine(f'sqlite:///{self.db_file}')
        self.data = pd.read_sql('models', engine)

    def __str__(self):
        return f"SdbGrid based on '{self.db_file}' database and with models located at '{self.grid_dir}'"

    def __repr__(self):
        return f"SdbGrid(db_file={self.db_file}, grid_dir={self.grid_dir})"

    def read_history(self, log_dir: str, top_dir: str, he4: float,
                     dest_dir: str = '.', delete_file: bool = True,
                     rename: bool = False, keep_tree: bool = False) -> mesa.MesaData:
        """Reads a single evolutionary model (a profile) and returns
        a MesaData object.

        Parameters
        ----------
        log_dir : str
            Log directory.
        top_dir : str
            Top directory.
        he4 : float
            Central helium abundance of the required model.
        dest_dir : str, optional
            Temporary dirctory for the required model. Default: '.'.
        delete_file : bool, optional
            If True delete the extracted model. The model is not deleted
            if 'keep_tree' is True. Default: True.
        rename : bool, optional
            If True it renames the history file to include information about
            the model contained in log_dir.
        keep_tree : bool, optional
            If True extract file with its directory structure (default
            ZipFile.extract behaviour), otherwise extract file directly to
            'dest_dir'. Default: False.

        Returns
        ----------
        MesaData
            Evolutionary model (MESA profile file) as MesaData object.
        """

        history_name = f'history{log_dir[4:]}.data' if rename else 'history.data'
        if keep_tree:
            file_name = os.path.join(
                dest_dir, top_dir, log_dir, self.evol_model_name(he4))
        else:
            file_name = os.path.join(dest_dir, history_name)
        if not self.model_extracted(file_name):
            self.extract_history(log_dir, top_dir, dest_dir, rename, keep_tree)
        data = mesa.MesaData(file_name)
        if delete_file and not keep_tree:
            os.remove(file_name)
        return data

    def read_evol_model(self, log_dir: str, top_dir: str, he4: float,
                        dest_dir: str = '.', delete_file: bool = True,
                        keep_tree: bool = False) -> mesa.MesaData:
        """Reads a single evolutionary model (a profile) and returns
        a MesaData object.

        Parameters
        ----------
        log_dir : str
            Log directory.
        top_dir : str
            Top directory.
        he4 : float
            Central helium abundance of the required model.
        dest_dir : str, optional
            Dirctory for the required model. Default: '.'.
        delete_file : bool, optional
            If True delete the extracted model. The model is not deleted
            if 'keep_tree' is True. Default: True.
        keep_tree : bool, optional
            If True extract file with its directory structure (default
            ZipFile.extract behaviour), otherwise extract file directly to
            'dest_dir'. Default: False.

        Returns
        ----------
        MesaData
            Evolutionary model (MESA profile file) as MesaData object.
        """

        if keep_tree:
            file_name = os.path.join(
                dest_dir, top_dir, log_dir, self.evol_model_name(he4))
        else:
            file_name = os.path.join(dest_dir, self.evol_model_name(he4))
        if not self.model_extracted(file_name):
            self.extract_evol_model(log_dir, top_dir, he4, dest_dir, keep_tree)
        data = mesa.MesaData(file_name)
        if delete_file and not keep_tree:
            os.remove(file_name)
        return data

    def read_puls_model(self, log_dir: str, top_dir: str, he4: float,
                        dest_dir: str = '.', delete_file: str = True,
                        keep_tree=False) -> GyreData:
        """Reads a calculated GYRE model and returns
        a GyreData object.

        Parameters
        ----------
        log_dir : str
            Log directory.
        top_dir : str
            Top directory.
        he4 : float
            Central helium abundance of the required model.
        dest_dir : str, optional
            Temporary dirctory for the required model. Default: '.'.
        delete_file : bool, optional
            If True delete the extracted model. The model is not deleted
            if 'keep_tree' is True. Default: True.
        keep_tree : bool, optional
            If True extract file with its directory structure (default
            ZipFile.extract behaviour), otherwise extract file directly to
            'dest_dir'. Default: False.

        Returns
        ----------
        GyreData
            Pulsation model as GyreData object.
        """

        if keep_tree:
            file_name = os.path.join(
                dest_dir, top_dir, log_dir, self.puls_model_name(he4))
        else:
            file_name = os.path.join(dest_dir, self.puls_model_name(he4))
        if not self.model_extracted(file_name):
            self.extract_puls_model(log_dir, top_dir, he4, dest_dir, keep_tree)
        data = GyreData(file_name)
        if delete_file and not keep_tree:
            os.remove(file_name)
        return data

    def extract_history(self, log_dir: str, top_dir: str,
                        dest_dir: str, rename: bool = False,
                        keep_tree: bool = False):
        """Extracts a MESA history file.

        Parameters
        ----------
        log_dir : str
            Log directory.
        top_dir : str
            Top directory.
        dest_dir : str
            Destination directory for the extracted model if 'keep_tree' is False,
            or the root direcotry if 'keep_tree' is True.
        rename : bool
            If True it renames the history file to include information about
            the model contained in log_dir. Default: False.
        keep_tree : bool, optional
            If True extract file with its directory structure (default
            ZipFile.extract behaviour), otherwise extract file directly to
            'dest_dir'. Default: False.

        Returns
        ----------
        """

        grid_zip_file = os.path.join(self.grid_dir, self.archive_name(top_dir))
        history_name = f'history{log_dir[4:]}.data' if rename else 'history.data'
        grid_zip_path = os.path.join(top_dir, log_dir, history_name)
        dest_path = os.path.join(dest_dir, history_name)

        with ZipFile(grid_zip_file) as archive:
            if keep_tree:
                archive.extract(grid_zip_path, dest_dir)
            else:
                with archive.open(grid_zip_path) as zipped_file, open(dest_path, 'wb') as dest_file:
                    shutil.copyfileobj(zipped_file, dest_file)

    def extract_evol_model(self, log_dir: str, top_dir: str, he4: float,
                           dest_dir: str, keep_tree: bool = False):
        """Extracts a single evolutionary model (a profile).

        Parameters
        ----------
        log_dir : str
            Log directory.
        top_dir : str
            Top directory.
        he4 : float
            Central helium abundance of the required model.
        dest_dir : str
            Destination directory for the extracted model if 'keep_tree' is False,
            or the root direcotry if 'keep_tree' is True.
        keep_tree : bool, optional
            If True extract file with its directory structure (default
            ZipFile.extract behaviour), otherwise extract file directly to
            'dest_dir'. Default: False.

        Returns
        ----------
        """

        grid_zip_file = os.path.join(self.grid_dir, self.archive_name(top_dir))
        model_name = self.evol_model_name(he4)
        grid_zip_path = os.path.join(top_dir, log_dir, model_name)
        dest_path = os.path.join(dest_dir, model_name)

        with ZipFile(grid_zip_file) as archive:
            if grid_zip_path in archive.namelist():
                if keep_tree:
                    archive.extract(grid_zip_path, dest_dir)
                else:
                    with archive.open(grid_zip_path) as zipped_file, open(dest_path, 'wb') as dest_file:
                        shutil.copyfileobj(zipped_file, dest_file)

    def extract_puls_model(self, log_dir: str, top_dir: str, he4: float,
                           dest_dir: str, keep_tree: bool = False):
        """Extracts a single calculated GYRE model.

        Parameters
        ----------
        log_dir : str
            Log directory.
        top_dir : str
            Top directory.
        he4 : float
            Central helium abundance of the required model.
        dest_dir : str
            Destination directory for the extracted model if 'keep_tree' is False,
            or the root direcotry if 'keep_tree' is True.
        keep_tree : bool, optional
            If True extract file with its directory structure (default
            ZipFile.extract behaviour), otherwise extract file directly to
            'dest_dir'. Default: False.

        Returns
        ----------
        """

        grid_zip_file = os.path.join(self.grid_dir, self.archive_name(top_dir))
        model_name = self.puls_model_name(he4)
        grid_zip_path = os.path.join(top_dir, log_dir, model_name)
        dest_path = os.path.join(dest_dir, model_name)

        with ZipFile(grid_zip_file) as archive:
            if grid_zip_path in archive.namelist():
                if keep_tree:
                    archive.extract(grid_zip_path, dest_dir)
                else:
                    with archive.open(grid_zip_path) as zipped_file, open(dest_path, 'wb') as dest_file:
                        shutil.copyfileobj(zipped_file, dest_file)

    def extract_gyre_input_model(self, log_dir: str, top_dir: str, he4: float,
                                 dest_dir: str, keep_tree: bool = False):
        """Extracts a single GYRE input model.

        Parameters
        ----------
        log_dir : str
            Log directory.
        top_dir : str
            Top directory.
        he4 : float
            Central helium abundance of the required model.
        dest_dir : str
            Destination directory for the extracted file if 'keep_tree' is False,
            or the root direcotry if 'keep_tree' is True.
        keep_tree : bool, optional
            If True extract file with its directory structure (default
            ZipFile.extract behaviour), otherwise extract file directly to
            'dest_dir'. Default: False.

        Returns
        ----------
        """

        grid_zip_file = os.path.join(self.grid_dir, self.archive_name(top_dir))
        model_name = self.gyre_input_name(he4)
        grid_zip_path = os.path.join(top_dir, log_dir, model_name)
        dest_path = os.path.join(dest_dir, model_name)

        with ZipFile(grid_zip_file) as archive:
            if grid_zip_path in archive.namelist():
                if keep_tree:
                    archive.extract(grid_zip_path, dest_dir)
                else:
                    with archive.open(grid_zip_path) as zipped_file, open(dest_path, 'wb') as dest_file:
                        shutil.copyfileobj(zipped_file, dest_file)

    def extract_log_dir(self, log_dir: str, top_dir: str, dest_dir: str):
        """Extracts a MESA log directory.

        Parameters
        ----------
        log_dir : str
            Log directory.
        top_dir : str
            Top directory.
        dest_dir : str
            Destination directory for the extracted directory tree.

        Returns
        ----------
        """

        grid_zip_file = os.path.join(self.grid_dir, self.archive_name(top_dir))
        grid_zip_path = os.path.join(top_dir, log_dir)

        with ZipFile(grid_zip_file) as archive:
            for f_name in archive.namelist():
                if f_name.startswith(grid_zip_path):
                    archive.extract(f_name, dest_dir)

    def evol_model_exists(self, log_dir: str, top_dir: str, he4: float) -> bool:
        """Checks if a profile exists in archive.

        Parameters
        ----------
        log_dir : str
            Log directory.
        top_dir : str
            Top directory.
        he4 : float
            Central helium abundance of the required model.

        Returns
        ----------
        bool
            True if a profile exists, False otherwise.
        """

        grid_zip_file = os.path.join(self.grid_dir, self.archive_name(top_dir))
        model_name = self.evol_model_name(he4)
        grid_zip_path = os.path.join(top_dir, log_dir, model_name)

        with ZipFile(grid_zip_file) as archive:
            if grid_zip_path in archive.namelist():
                return True
            else:
                return False

    def puls_model_exists(self, log_dir: str, top_dir: str, he4: float) -> bool:
        """Checks if a calculated GYRE model exists in archive.

        Parameters
        ----------
        log_dir : str
            Log directory.
        top_dir : str
            Top directory.
        he4 : float
            Central helium abundance of the required model.

        Returns
        ----------
        bool
            True if a profile exists, False otherwise.
        """

        grid_zip_file = os.path.join(self.grid_dir, self.archive_name(top_dir))
        model_name = self.puls_model_name(he4)
        grid_zip_path = os.path.join(top_dir, log_dir, model_name)

        with ZipFile(grid_zip_file) as archive:
            if grid_zip_path in archive.namelist():
                return True
            else:
                return False

    def gyre_input_exists(self, log_dir: str, top_dir: str, he4: float) -> bool:
        """Checks if a GYRE input model exists in archive.

        Parameters
        ----------
        log_dir : str
            Log directory.
        top_dir : str
            Top directory.
        he4 : float
            Central helium abundance of the required model.

        Returns
        ----------
        bool
            True if a profile exists, False otherwise.
        """

        grid_zip_file = os.path.join(self.grid_dir, self.archive_name(top_dir))
        model_name = self.gyre_input_name(he4)
        grid_zip_path = os.path.join(top_dir, log_dir, model_name)

        with ZipFile(grid_zip_file) as archive:
            if grid_zip_path in archive.namelist():
                return True
            else:
                return False

    def df_from_errorbox(self, star: Star, sigma: float = 1.0,
                         use_teff: bool = True, use_logg: bool = True,
                         use_vrot: bool = False, use_feh: bool = False,
                         use_z_surf: bool = False) -> DataFrame:
        """Selects models based on the observational
        parameters of a star.

        Parameters
        ----------
        star : Star
            A star for which models are selected.
        sigma : float, optional
            Size of the considered error box expressed
            as a multiplier of error.
            Default: 1.0.
        use_teff : bool, optional
            If True uses effective temperature for selection.
            Default: True.
        use_logg : bool, optional
            If True uses log_g for selection.
            Default: True.
        use_vrot : bool, optional
            If True uses rotational velocity for selection.
            Default: False.
        use_feh : bool, optional
            If True uses metallicity for selection.
            Default: False.
        use_z_surf : bool, optional
            If True uses surface Z for selection of [Fe/H], otherwise
            uses initial Z of progenitor.
            Default: False.

        Returns
        ----------
        DaraFrame
            Dataframe containing the selected models.
        """

        c = True

        if use_teff:
            c_teff = (10.0 ** self.data.log_Teff <= star.t_eff + sigma*star.t_eff_err_p) & \
                (10.0 ** self.data.log_Teff >= star.t_eff - sigma*star.t_eff_err_m)
            c &= c_teff

        if use_logg:
            c_logg = (self.data.log_g <= star.log_g + sigma*star.log_g_err_p) & \
                (self.data.log_g >= star.log_g - sigma*star.log_g_err_m)
            c &= c_logg

        if use_vrot:
            c_vrot = (self.data.rot <= star.v_rot + sigma*star.v_rot_err_p) & \
                (self.data.rot >= star.v_rot - sigma*star.v_rot_err_m)
            c &= c_vrot

        if use_feh:
            if use_z_surf:
                c_feh = (self.calc_feh(self.data.z_surf) <= star.feh + sigma*star.feh_err_p) & \
                    (self.calc_feh(self.data.z_surf) >=
                     star.feh - sigma*star.feh_err_m)
            else:
                c_feh = (self.calc_feh(self.data.z_i) <= star.feh + sigma*star.feh_err_p) & \
                    (self.calc_feh(self.data.z_i) >=
                     star.feh - sigma*star.feh_err_m)
            c &= c_feh

        return self.data[c]

    @staticmethod
    def model_extracted(path: str) -> bool:
        """Checks if model is already exracted.

        Parameters
        ----------
        path : str
            Path to the model.

        Returns
        ----------
        bool
            True if model exists, False otherwise.
        """

        if os.path.isfile(path):
            return True
        else:
            return False

    @staticmethod
    def archive_name(top_dir: str) -> str:
        """Returns a name of a zip file containing a top direcotry.

        Parameters
        ----------
        top_dir : str
            Top directory.

        Returns
        ----------
        str
            Name of zipfile.
        """

        return f"grid{top_dir[4:]}.zip"

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

        return f"custom_He{round(he4, 6)}.data"

    @staticmethod
    def puls_model_name(he4: float) -> str:
        """Returns a name of a calculated GYRE model for helium abundance 'he4'.

        Parameters
        ----------
        he4 : float
            Central helium abundance of the model.

        Returns
        ----------
        str
            Name of calculated GYRE model.
        """

        return f"custom_He{round(he4, 6)}_summary.txt"

    @staticmethod
    def gyre_input_name(he4: float) -> str:
        """Returns a name of an input model for GYRE model for helium abundance 'he4'.

        Parameters
        ----------
        he4 : float
            Central helium abundance of the model.

        Returns
        ----------
        str
            Name of GYRE input model.
        """

        return f"custom_He{round(he4, 6)}.data.GYRE"

    @staticmethod
    def calc_feh(z: float) -> float:
        """Calculates [Fe/H] from metallicity.
        Assumes solar chemical compostion from Asplund et al. (2009).

        Parameters
        ----------
        z : float
            Metallicity.

        Returns
        ----------
        float
            Calculated [Fe/H].
        """

        solar_h1 = 0.7154
        solar_h2 = 1.43e-5
        solar_he3 = 4.49e-5
        solar_he4 = 0.2702551

        solar_x = solar_h1 + solar_h2
        solar_y = solar_he3 + solar_he4
        solar_z = 1.0 - solar_x - solar_y

        return np.log10(z/solar_z)
