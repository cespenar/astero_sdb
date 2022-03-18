from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
from pandas.core.frame import DataFrame
from tqdm import tqdm

from .sdb_grid_reader import SdbGrid


class Star:
    """Structure containing observational properties of a star.

    """

    num_of_stars = 0

    def __init__(self,
                 name: str,
                 t_eff: float = None,
                 t_eff_err_p: float = None,
                 t_eff_err_m: float = None,
                 log_g: float = None,
                 log_g_err_p: float = None,
                 log_g_err_m: float = None,
                 v_rot: float = None,
                 v_rot_err_p: float = None,
                 v_rot_err_m: float = None,
                 feh: float = None,
                 feh_err_p: float = None,
                 feh_err_m: float = None,
                 luminosity: float = None,
                 luminosity_err_p: float = None,
                 luminosity_err_m: float = None,
                 rad: float = None,
                 rad_err_p: float = None,
                 rad_err_m: float = None,
                 frequencies_list: Union[str, Path] = None):
        """Creates a Star object using provided observational data.

        Parameters
        ----------
        name : str
            Name of the star.
        t_eff : float, optional
            Effective temperature. Default: None.
        t_eff_err_p : float, optional
            Plus-error of effective temperature. Default: None.
        t_eff_err_m : float, optional
            Minus-error of effective temperature. Default: None.
        log_g : float, optional
            Surface log(g). Default: None.
        log_g_err_p : float, optional
            Plus-error of log(g). Default: None.
        log_g_err_m : float, optional
            Minus-error of log(g). Default: None.
        v_rot : float, optional
            Surface rotational velocity. Default: None.
        v_rot_err_p : float, optional
            Plus-error of v_rot. Default: None.
        v_rot_err_m : float, optional
            Minus-error of v_rot. Default: None.
        feh : float, optional
            Surface metallicity [Fe/H]. Default: None.
        feh_err_p : float, optional
            Plus-error of metallicity. Default: None.
        feh_err_m : float, optional
            Minus-error of metallicity. Default: None.
        luminosity : float, optional
            Luminosity. Default: None.
        luminosity_err_p : float, optional
            Plus-error of L. Default: None.
        luminosity_err_m : float, optional
            Minus-error of L. Default: None.
        rad : float, optional
            Radius in solar units. Default: None.
        rad_err_p : float, optional
            Plus-error of radius. Default: None.
        rad_err_m : float, optional
            Minus-error of radius. Default: None.
        frequencies_list : Union[str, Path], optional
            Text file containing list of observed frequencies.
            Default: None.
        """

        self.name = name
        self.t_eff = t_eff
        self.t_eff_err_p = t_eff_err_p
        self.t_eff_err_m = t_eff_err_m
        self.log_g = log_g
        self.log_g_err_p = log_g_err_p
        self.log_g_err_m = log_g_err_m
        self.v_rot = v_rot
        self.v_rot_err_p = v_rot_err_p
        self.v_rot_err_m = v_rot_err_m
        self.feh = feh
        self.feh_err_p = feh_err_p
        self.feh_err_m = feh_err_m
        self.luminosity = luminosity
        self.luminosity_err_p = luminosity_err_p
        self.luminosity_err_m = luminosity_err_m
        self.rad = rad
        self.rad_err_p = rad_err_p
        self.rad_err_m = rad_err_m
        if frequencies_list:
            self.frequencies = np.genfromtxt(frequencies_list, dtype=None,
                                             skip_header=1, names=True)
        else:
            self.frequencies = None

        Star.num_of_stars += 1

    def __str__(self):
        return f'{self.name}'

    def __repr__(self):
        return (
            f'Star({self.name}, '
            f't_eff={self.t_eff}, t_eff_err_p={self.t_eff_err_p}, '
            f't_eff_err_m={self.t_eff_err_m}, '
            f'log_g={self.log_g}, log_g_err_p={self.log_g_err_p}, '
            f'log_g_err_m={self.log_g_err_m}, '
            f'v_rot={self.v_rot}, v_rot_err_p={self.v_rot_err_p}, '
            f'v_rot_err_m={self.v_rot_err_m}, '
            f'feh={self.feh}, feh_err_p={self.feh_err_p}, '
            f'feh_err_m={self.feh_err_m}, '
            f'luminosity={self.luminosity}, '
            f'luminosity_err_p={self.luminosity_err_p}, '
            f'luminosity_err_m={self.luminosity_err_m}, '
            f'rad={self.rad}, rad_err_p={self.rad_err_p}, '
            f'rad_err_m={self.rad_err_m}, '
            f'frequencies_list={self.frequencies})'
        )

    def unique_multiplet_ids(self) -> np.ndarray:
        """Returns list of multiplet indices.

        Returns
        -------
        numpy.array
            Numpy array with unique multiplet indices.
        """
        return np.unique(
            self.frequencies['idm'][~np.isnan(self.frequencies['idm'])])

    def period_combinations(self) -> list[dict]:
        """Finds all possible combinations of periods for identified triplets
        and doublets.

        Returns
        -------
        list[dict]
            List of dictionaries containing combinations of periods.
            Supplemented with ID and l.
        """

        periods = [{}, ]
        id_multiplets = self.unique_multiplet_ids()
        for id in id_multiplets:
            df_multi = self.frequencies[self.frequencies['idm'] == id]
            deg = df_multi['l'][0]
            if len(df_multi) == 3:
                for p_dict in periods:
                    p_dict[df_multi['id'][1]] = {'P': df_multi['P'][1],
                                                 'l': deg}
            if len(df_multi) == 2:
                if (df_multi['m'][0] == -1) and (df_multi['m'][1] == 1):
                    for p_dict in periods:
                        id_middle = round(
                            (df_multi['id'][0] + df_multi['id'][1]) / 2.0, 1)
                        p_middle = round(
                            (df_multi['P'][0] + df_multi['P'][1]) / 2.0, 5)
                        p_dict[id_middle] = {'P': p_middle, 'l': deg}
                else:
                    periods_temp = []
                    for p_dict in periods:
                        p_dict_temp = deepcopy(p_dict)
                        p_dict[df_multi['id'][0]] = {
                            'P': df_multi['P'][0], 'l': deg}
                        p_dict_temp[df_multi['id'][1]] = {
                            'P': df_multi['P'][1], 'l': deg}
                        periods_temp.append(p_dict_temp)
                    for p_dict in periods_temp:
                        periods.append(p_dict)
            if len(df_multi) == 1:
                for p_dict in periods:
                    p_dict[df_multi['id'][0]] = {'P': df_multi['P'][0],
                                                 'l': deg}
        return periods

    def periods_explicit(self) -> list[dict]:
        """Lists all identified periods explicitly taking into account values
        of m provided in the list of frequencies and returns them in format
        compatible with self.period_combinations().

        If a selected period is an average of two periods, the id of
        a component with a negative m is used.

        Returns
        -------
        list[dict]
            List of dictionaries containing periods, supplemented with ID
            and l.
        """

        periods = {}
        id_multiplets = self.unique_multiplet_ids()
        for id in id_multiplets:
            df_multi = self.frequencies[self.frequencies['idm'] == id]
            deg = df_multi['l'][0]
            if 'm' in self.frequencies.dtype.names and not np.isnan(
                    self.frequencies['m']).all():
                if 0 in df_multi['m']:
                    i = df_multi['m'].tolist().index(0)
                    periods[df_multi['id'][i]] = {'P': df_multi['P'][i],
                                                  'l': deg}
                else:
                    for m in np.sort(np.unique(np.abs(df_multi['m']))):
                        if -m in df_multi['m'] and m in df_multi['m']:
                            im = df_multi['m'].tolist().index(-m)
                            ip = df_multi['m'].tolist().index(m)
                            p_middle = round(
                                (df_multi['P'][im] + df_multi['P'][ip]) / 2.0,
                                5)
                            periods[df_multi['id'][im]] = {'P': p_middle,
                                                           'l': deg}
                            break
            else:
                periods[df_multi['id'][0]] = {'P': df_multi['P'][0],
                                              'l': deg}
        return [periods]

    def chi2_star(self,
                  df_selected: DataFrame,
                  use_z_surf: bool = True) -> None:
        """Calculates chi^2 function for the star and models provided in the
        given grid. Utilizes available global stellar parameters.

        Parameters
        ----------
        df_selected : pandas.DataFrame
            Pandas DataFrame containing the grid.
        use_z_surf : bool, optional
            If True uses surface Z for selection of [Fe/H], otherwise uses
            initial Z of progenitor. Default: True.

        Returns
        -------

        """
        df_selected['chi2_star'] = 0.0

        if self.t_eff:
            df_selected.chi2_star += self.chi2_single(
                x_model=10.0 ** df_selected.log_Teff,
                x_obs=self.t_eff,
                sigma=self.t_eff_err_p
            )

        if self.log_g:
            df_selected.chi2_star += self.chi2_single(
                x_model=df_selected.log_g,
                x_obs=self.log_g,
                sigma=self.log_g_err_p
            )

        if self.v_rot:
            df_selected.chi2_star += self.chi2_single(x_model=df_selected.rot,
                                                      x_obs=self.v_rot,
                                                      sigma=self.v_rot_err_p
                                                      )

        if self.feh:
            if use_z_surf:
                df_selected.chi2_star += self.chi2_single(
                    x_model=df_selected.z_surf,
                    x_obs=self.feh,
                    sigma=self.feh_err_p
                )
            else:
                df_selected.chi2_star += self.chi2_single(
                    x_model=self.calc_feh(df_selected.z_i),
                    x_obs=self.feh,
                    sigma=self.feh_err_p
                )

    def chi2_puls(self,
                  df_selected: DataFrame,
                  grid: SdbGrid,
                  dest_dir: Path,
                  ignore_combinations: bool = True,
                  ignore_degree: bool = False,
                  max_deg_if_ignore: int = 2,
                  save_period_list: bool = False,
                  period_list_name: str = None,
                  progress: bool = True) -> None:
        """Calculates chi^2 function for the star and a grid using available
        pulsation periods.

        Parameters
        ----------
        df_selected : pandas.DataFrame
            Pandas DataFrame containing the models selected for chi^2
            calculation.
        grid : SdbGrid
            Complete grid of sdB models.
        dest_dir : Path
            Target root directory for extracted models.
        ignore_combinations : bool, optional
            If True ignores potential combinations of periods due to missing
            components of multiplets. Default: True.
        ignore_degree : bool, optional
            If True ignore mode identification during chi^2 minimization.
            Default: False.
        max_deg_if_ignore : int, optianal
            Maximum degree used for minimizing chi^2 if ignore_degree is True.
            Default: 2.
        save_period_list : bool, optional
            If True creates a file with listed all combinations of periods used
            to calculate chi^2 function.
        period_list_name : str, optional
            Name of output file saved when save_period_list is True. If None
            default name is used. Default: None.
        progress: bool, optional
            If true shows a progress bar. Default: True.

        Returns
        -------

        """

        if ignore_combinations:
            period_combinations = self.periods_explicit()
        else:
            period_combinations = self.period_combinations()

        if save_period_list:
            if period_list_name:
                f_name = Path(period_list_name)
            else:
                f_name = Path(f'{self.name}_periods.txt')

            with f_name.open(mode='w') as f:
                f.write(f'{self.name}\n')
                f.write(f'{len(period_combinations)} period combinations\n\n')
                for i, p_dict in enumerate(period_combinations):
                    f.write(f'--- puls_{i + 1} ---\n')
                    for id, p in p_dict.items():
                        f.write(
                            f'ID: {id:4}, P: {p["P"]:12}, l: {int(p["l"]):1}\n')
                    f.write('\n')

        for i in range(len(period_combinations)):
            df_selected[f'chi2_puls_{i + 1}'] = 0.0

        if progress:
            pbar = tqdm(total=len(df_selected))
        for index, model in df_selected.iterrows():
            puls_data = grid.read_puls_model(log_dir=model.log_dir,
                                             top_dir=model.top_dir,
                                             he4=model.custom_profile,
                                             dest_dir=dest_dir,
                                             delete_file=False,
                                             keep_tree=True)
            if ignore_degree:
                combined_periods = puls_data.periods(deg=1, g_modes_only=True)
                if max_deg_if_ignore > 1:
                    for d in range(2, max_deg_if_ignore + 1):
                        combined_periods = np.concatenate((combined_periods,
                                                           puls_data.periods(
                                                               deg=d,
                                                               g_modes_only=True)))
                combined_periods = np.sort(combined_periods)[::-1]
            for i, periods in enumerate(period_combinations):
                chi2 = 0.0
                for p_obs in periods.values():
                    if ignore_degree:
                        delta = np.min(np.abs(combined_periods - p_obs['P']))
                    else:
                        delta = np.min(np.abs(puls_data.periods(
                            p_obs['l'], g_modes_only=True) - p_obs['P']))
                chi2 += delta ** 2.0
                chi2 /= len(periods)
                df_selected[f'chi2_puls_{i + 1}'][index] = chi2
            if progress:
                pbar.set_description('Calculating chi^2 puls')
                pbar.update(1)
        if progress:
            pbar.close()

    def evaluate_chi2(self,
                      df_selected: DataFrame,
                      grid: SdbGrid,
                      dest_dir: Path,
                      use_spectroscopy: bool = True,
                      use_periods: bool = True,
                      ignore_combinations: bool = True,
                      ignore_degree: bool = False,
                      max_deg_if_ignore: int = 2,
                      save_period_list: bool = False,
                      period_list_name: str = None,
                      progress: bool = True,
                      use_z_surf: bool = True,
                      save_results: bool = True,
                      results_file_name: str = None) -> None:
        """Evaluates chi^2 functions for the star.

        Parameters
        ----------
        df_selected : pandas.DataFrame
            Pandas DataFrame containing the models selected for chi^2
            calculation.
        grid : SdbGrid
            Complete grid of sdB models.
        dest_dir : Path
            Target root directory for extracted models.
        use_spectroscopy : bool, optional
            If True calculates chi^2 using available spectroscopic parameters.
            Default: True.
        use_periods : bool, optional
            If True calculates chi^2 using available pulsational periods.
            Default: True.
        ignore_combinations : bool, optional
            If True ignores potential combinations of periods due to missing
            components of multiplets. Default: True.
        ignore_degree : bool, optional
            If True ignore mode identification during chi^2 minimization using
            available pulsational periods. Default: False.
        max_deg_if_ignore : int, optional
            Maximum degree used for minimizing chi^2 if ignore_degree is True.
            Default: 2.
        save_period_list : bool, optional
            If True creates a file with listed all combinations of periods used
            to calculate chi^2 function.
        period_list_name : str, optional
            Name of output file saved when save_period_list is True. If None
            default name is used. Default: None.
        progress: bool, optional
            If true shows a progress bar. Default: True.
        use_z_surf : bool, optional
            If True uses surface Z for selection of [Fe/H], otherwise uses
            initial Z of progenitor. Default: True.
        save_results : bool, optional
            If True saves the DataFrame containing calculated values of chi^2
            to a text file. Default: True.
        results_file_name : str, optional
            Name of the output file containing values of chi^2. If not provided
            default name is used. Default: None.

        Returns
        -------

        """

        if use_spectroscopy:
            self.chi2_star(df_selected=df_selected, use_z_surf=use_z_surf)
        if use_periods:
            self.chi2_puls(df_selected=df_selected,
                           grid=grid,
                           dest_dir=dest_dir,
                           ignore_combinations=ignore_combinations,
                           ignore_degree=ignore_degree,
                           max_deg_if_ignore=max_deg_if_ignore,
                           save_period_list=save_period_list,
                           period_list_name=period_list_name,
                           progress=progress)
        if save_results:
            if results_file_name:
                f_name = Path(results_file_name)
            else:
                f_name = Path(f'{self.name}_chi2.txt')
            df_selected.to_csv(f_name, sep=' ', header=True, index=False)

    def df_from_errorbox(self,
                         grid: SdbGrid,
                         sigma: float = 1.0,
                         use_teff: bool = True,
                         use_logg: bool = True,
                         use_vrot: bool = False,
                         use_feh: bool = False,
                         use_z_surf: bool = False) -> DataFrame:
        """Selects models from a grid based on the observational parameters of
        the star.

        Parameters
        ----------
        grid : SdbGrid
            A grid of sdB stars.
        sigma : float, optional
            Size of the considered error box expressed as a multiplier of
            error. Default: 1.0.
        use_teff : bool, optional
            If True uses effective temperature for selection. Default: True.
        use_logg : bool, optional
            If True uses log_g for selection. Default: True.
        use_vrot : bool, optional
            If True uses rotational velocity for selection. Default: False.
        use_feh : bool, optional
            If True uses metallicity for selection. Default: False.
        use_z_surf : bool, optional
            If True uses surface Z for selection of [Fe/H], otherwise uses
            initial Z of progenitor. Default: False.

        Returns
        ----------
        DaraFrame
            Dataframe containing the selected models.
        """

        c = True

        if use_teff:
            c_teff = (10.0 ** grid.data.log_Teff <= self.t_eff
                      + sigma * self.t_eff_err_p) & \
                     (10.0 ** grid.data.log_Teff >= self.t_eff
                      - sigma * self.t_eff_err_m)
            c &= c_teff

        if use_logg:
            c_logg = (grid.data.log_g <= self.log_g
                      + sigma * self.log_g_err_p) & \
                     (grid.data.log_g >= self.log_g - sigma * self.log_g_err_m)
            c &= c_logg

        if use_vrot:
            c_vrot = (grid.data.rot <= self.v_rot + sigma * self.v_rot_err_p) & \
                     (grid.data.rot >= self.v_rot - sigma * self.v_rot_err_m)
            c &= c_vrot

        if use_feh:
            if use_z_surf:
                c_feh = (self.calc_feh(
                    grid.data.z_surf) <= self.feh + sigma * self.feh_err_p) & \
                        (self.calc_feh(grid.data.z_surf) >=
                         self.feh - sigma * self.feh_err_m)
            else:
                c_feh = (self.calc_feh(
                    grid.data.z_i) <= self.feh + sigma * self.feh_err_p) & \
                        (self.calc_feh(grid.data.z_i) >=
                         self.feh - sigma * self.feh_err_m)
            c &= c_feh

        return grid.data[c]

    @staticmethod
    def calc_feh(z: float) -> float:
        """Calculates [Fe/H] from metallicity.
        Assumes solar chemical composition from Asplund et al. (2009).

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

        return np.log10(z / solar_z)

    @staticmethod
    def chi2_single(x_model: np.array,
                    x_obs: float,
                    sigma: float) -> np.ndarray:
        """Calculates a single component of chi^2
        function.

        Parameters
        ----------
        x_model :
            Modelled values.
        x_obs : float
            Observed value.
        sigma : float
            Observational error.

        Returns
        -------
        numpy.ndarray
            A single component of chi^2 function.
        """

        return ((x_obs - x_model) / sigma) ** 2.0
