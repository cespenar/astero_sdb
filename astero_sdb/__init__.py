import numpy as np
from pandas.core.frame import DataFrame

from astero_sdb.sdb_grid_reader import SdbGrid
from astero_sdb.star import Star


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


def chi2_star(star: Star, grid: SdbGrid, use_z_surf: bool = True):
    """Calculates chi^2 function for a given star
    and models provided in the given grid. Utilizes
    availalbe global stellar parameters.

    Parameters
    ----------
    star : Star
        A star for which chi^2 function is calculated.
    grid : pandas.DataFrame
        Pandas DataFrame containing the grid.
    use_z_surf : bool, optional
        If True uses surface Z for selection of [Fe/H],
        otherwise uses initial Z of progenitor.
        Default: True.

    Returns
    -------

    """
    grid['chi2_star'] = 0.0

    if star.t_eff:
        grid.chi2_star += chi2_single(x_model=10.0 ** grid.log_Teff,
                                      x_obs=star.t_eff,
                                      sigma=star.t_eff_err_p
                                      )

    if star.log_g:
        grid.chi2_star += chi2_single(x_model=grid.log_g,
                                      x_obs=star.log_g,
                                      sigma=star.log_g_err_p
                                      )

    if star.v_rot:
        grid.chi2_star += chi2_single(x_model=grid.rot,
                                      x_obs=star.v_rot,
                                      sigma=star.v_rot_err_p
                                      )

    if star.feh:
        if use_z_surf:
            grid.chi2_star += chi2_single(x_model=calc_feh(grid.z_surf),
                                          x_obs=star.feh,
                                          sigma=star.feh_err_p
                                          )
        else:
            grid.chi2_star += chi2_single(x_model=calc_feh(grid.z_i),
                                          x_obs=star.feh,
                                          sigma=star.feh_err_p
                                          )


def chi2_puls(star: Star, df_selected: DataFrame, grid: SdbGrid,
              dest_dir: str, save_period_list: bool = False,
              period_list_name: str = None):
    """Calculates chi^2 function for a star
    and a grid using availiable pulsation periods.

    Parameters
    ----------
    star : Star
        A star for which chi^2 function is calculated.
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

    Returns
    -------

    """

    period_combinations = star.period_combinations()

    if save_period_list:
        if period_list_name:
            f_name = period_list_name
        else:
            f_name = f'{star.name}_periods.txt'

        with open(f_name, 'w') as f:
            f.write(f'{star.name}\n')
            f.write(f'{len(period_combinations)} period combinations\n\n')
            for i, p_dict in enumerate(period_combinations):
                f.write(f'--- puls_{i+1} ---\n')
                for id, p in p_dict.items():
                    f.write(
                        f"ID: {id:4}, P: {p['P']:12}, l: {int(p['l']):1}\n")
                f.write('\n')

    for i in range(len(period_combinations)):
        df_selected[f'chi2_puls_{i+1}'] = 0.0

    for index, model in df_selected.iterrows():
        puls_data = grid.read_puls_model(log_dir=model.log_dir,
                                         top_dir=model.top_dir,
                                         he4=model.custom_profile,
                                         dest_dir=dest_dir,
                                         delete_file=False,
                                         keep_tree=True)
        print(f'index: {index}')
        for i, periods in enumerate(period_combinations):
            chi2 = 0.0
            for p_obs in periods.values():
                delta = np.min(np.abs(puls_data.periods(
                    p_obs['l'], g_modes_only=True) - p_obs['P']))
                chi2 += delta ** 2.0
            chi2 /= len(periods)
            df_selected[f'chi2_puls_{i+1}'][index] = chi2


if __name__ == "__main__":
    pass
