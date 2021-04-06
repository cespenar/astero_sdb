from copy import deepcopy

import numpy as np
from pandas.core.frame import DataFrame
from tqdm import tqdm

from astero_sdb.sdb_grid_reader import SdbGrid


class Star:
    """Structure containing observational properties of a star.

    """

    num_of_stars = 0

    def __init__(self, name,
                 t_eff=None, t_eff_err_p=None, t_eff_err_m=None,
                 log_g=None, log_g_err_p=None, log_g_err_m=None,
                 v_rot=None, v_rot_err_p=None, v_rot_err_m=None,
                 feh=None, feh_err_p=None, feh_err_m=None,
                 frequencies_list=None):
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
        if frequencies_list:
            self.frequencies = np.genfromtxt(
                frequencies_list, dtype=None, skip_header=1, names=True)
        else:
            self.frequencies = None

        Star.num_of_stars += 1

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return (
            f"Star({self.name}, "
            f"t_eff={self.t_eff}, t_eff_err_p={self.t_eff_err_p}, t_eff_err_m={self.t_eff_err_m}, "
            f"log_g={self.log_g}, log_g_err_p={self.log_g_err_p}, log_g_err_m={self.log_g_err_m}, "
            f"v_rot={self.v_rot}, v_rot_err_p={self.v_rot_err_p}, v_rot_err_m={self.v_rot_err_m}, "
            f"feh={self.feh}, feh_err_p={self.feh_err_p}, feh_err_m={self.feh_err_m}, "
            f"frequencies_list={self.frequencies})"
        )

    def unique_multiplet_ids(self):
        """Returns list of multiplet indices.

        Returns
        -------
        numpy.array
            Numpy array with unique multiplet indices.
        """
        return np.unique(self.frequencies['idm'][~np.isnan(self.frequencies['idm'])])

    def period_combinations(self):
        """Finds all possible combinations of periods
        for identified triplets and doubles.

        Returns
        -------
        list[dict]
            List of dictionaries containing combinations
            of periods. Supplemented with ID and l.
        """

        periods = [{}, ]
        id_multiplets = self.unique_multiplet_ids()
        for id in id_multiplets:
            df_multi = self.frequencies[self.frequencies['idm'] == id]
            l = df_multi['l'][0]
            if len(df_multi) == 3:
                for p_dict in periods:
                    p_dict[df_multi['id'][1]] = {'P': df_multi['P'][1], 'l': l}
            else:
                if (df_multi['m'][0] == -1) and (df_multi['m'][1] == 1):
                    for p_dict in periods:
                        id_middle = round(
                            (df_multi['id'][0] + df_multi['id'][1])/2.0, 1)
                        p_middle = round(
                            (df_multi['P'][0] + df_multi['P'][1])/2.0, 5)
                        p_dict[id_middle] = {'P': p_middle, 'l': l}
                else:
                    periods_temp = []
                    for p_dict in periods:
                        p_dict_temp = deepcopy(p_dict)
                        p_dict[df_multi['id'][0]] = {
                            'P': df_multi['P'][0], 'l': l}
                        p_dict_temp[df_multi['id'][1]] = {
                            'P': df_multi['P'][1], 'l': l}
                        periods_temp.append(p_dict_temp)
                    for p_dict in periods_temp:
                        periods.append(p_dict)
        return periods

    def chi2_star(self, df_selected: DataFrame, use_z_surf: bool = True):
        """Calculates chi^2 function for the star
        and models provided in the given grid. Utilizes
        availalbe global stellar parameters.

        Parameters
        ----------
        star : Star
            A star for which chi^2 function is calculated.
        df_selected : pandas.DataFrame
            Pandas DataFrame containing the grid.
        use_z_surf : bool, optional
            If True uses surface Z for selection of [Fe/H],
            otherwise uses initial Z of progenitor.
            Default: True.

        Returns
        -------

        """
        df_selected['chi2_star'] = 0.0

        if self.t_eff:
            df_selected.chi2_star += self.chi2_single(x_model=10.0 ** df_selected.log_Teff,
                                                      x_obs=self.t_eff,
                                                      sigma=self.t_eff_err_p
                                                      )

        if self.log_g:
            df_selected.chi2_star += self.chi2_single(x_model=df_selected.log_g,
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
                df_selected.chi2_star += self.chi2_single(x_model=self(df_selected.z_surf),
                                                          x_obs=self.feh,
                                                          sigma=self.feh_err_p
                                                          )
            else:
                df_selected.chi2_star += self.chi2_single(x_model=self.calc_feh(df_selected.z_i),
                                                          x_obs=self.feh,
                                                          sigma=self.feh_err_p
                                                          )

    def chi2_puls(self, df_selected: DataFrame, grid: SdbGrid,
                  dest_dir: str, save_period_list: bool = False,
                  period_list_name: str = None, progress: bool = True):
        """Calculates chi^2 function for the star
        and a grid using availiable pulsation periods.

        Parameters
        ----------
        df_selected : pandas.DataFrame
            Pandas DataFrame containing the models selected
            for chi^2 calculation.
        grid : SdbGrid
            Complete grid of sdB models.
        dest_dir : str
            Target root directory for extracted models.
        save_period_list : bool, optional
            If True creates a file with listed all
            combinations of periods used to calculate chi^2
            fucntion.
        period_list_name : str, optional
            Name of output file saved when save_period_list
            is True. If None default name is used.
            Default: None.
        progress: bool, optional
            If true shows a progress bar. Default: True.

        Returns
        -------

        """

        period_combinations = self.period_combinations()

        if save_period_list:
            if period_list_name:
                f_name = period_list_name
            else:
                f_name = f'{self.name}_periods.txt'

            with open(f_name, 'w') as f:
                f.write(f'{self.name}\n')
                f.write(f'{len(period_combinations)} period combinations\n\n')
                for i, p_dict in enumerate(period_combinations):
                    f.write(f'--- puls_{i+1} ---\n')
                    for id, p in p_dict.items():
                        f.write(
                            f"ID: {id:4}, P: {p['P']:12}, l: {int(p['l']):1}\n")
                    f.write('\n')

        for i in range(len(period_combinations)):
            df_selected[f'chi2_puls_{i+1}'] = 0.0

        if progress:
            pbar = tqdm(total=len(df_selected))
        for index, model in df_selected.iterrows():
            puls_data = grid.read_puls_model(log_dir=model.log_dir,
                                             top_dir=model.top_dir,
                                             he4=model.custom_profile,
                                             dest_dir=dest_dir,
                                             delete_file=False,
                                             keep_tree=True)
            # print(f'index: {index}')
            for i, periods in enumerate(period_combinations):
                chi2 = 0.0
                for p_obs in periods.values():
                    delta = np.min(np.abs(puls_data.periods(
                        p_obs['l'], g_modes_only=True) - p_obs['P']))
                    chi2 += delta ** 2.0
                chi2 /= len(periods)
                df_selected[f'chi2_puls_{i+1}'][index] = chi2
            if progress:
                pbar.set_description('Calculating chi^2 puls')
                pbar.update(1)
        if progress:
            pbar.close()

    @staticmethod
    def calc_feh(z):
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

        return np.log10(z / solar_z)

    @staticmethod
    def chi2_single(x_model, x_obs, sigma):
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
        float
            A single component of chi^2 function.
        """

        return ((x_obs - x_model) / sigma) ** 2.0
