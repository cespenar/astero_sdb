import numpy as np

from .sdb_grid_reader import SdbGrid
from .star import Star


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


def chi2_star(star, df, use_z_surf=True):
    """Calculates chi^2 function for a given star
    and models provided in the given grid. Utilizes
    availalbe global stellar parameters.

    Parameters
    ----------
    star : Star
        A star for which chi^2 function is calculated.
    df : pandas.DataFrame
        Pandas DataFrame containing the grid.
    use_z_surf : bool, optional
        If True uses surface Z for selection of [Fe/H],
        otherwise uses initial Z of progenitor.
        Default: True.

    Returns
    -------

    """
    df['chi2_star'] = 0.0

    if star.t_eff:
        df.chi2_star += chi2_single(x_model=10.0 ** df.log_Teff,
                                    x_obs=star.t_eff,
                                    sigma=star.t_eff_err_p
                                    )

    if star.log_g:
        df.chi2_star += chi2_single(x_model=df.log_g,
                                    x_obs=star.log_g,
                                    sigma=star.log_g_err_p
                                    )

    if star.v_rot:
        df.chi2_star += chi2_single(x_model=df.rot,
                                    x_obs=star.v_rot,
                                    sigma=star.v_rot_err_p
                                    )

    if star.feh:
        if use_z_surf:
            df.chi2_star += chi2_single(x_model=calc_feh(star.z_surf),
                                        x_obs=star.feh,
                                        sigma=star.feh_err_p
                                        )
        else:
            df.chi2_star += chi2_single(x_model=calc_feh(star.z_i),
                                        x_obs=star.feh,
                                        sigma=star.feh_err_p
                                        )


if __name__ == "__main__":
    target = Star(name='KIC2991403',
                  t_eff=27300.0, t_eff_err_p=200.0, t_eff_err_m=200.0,
                  log_g=5.43, log_g_err_p=0.03, log_g_err_m=0.03,
                  frequencies_list='../../KIC2991403/KIC2991403_frequencies.txt')

    database = '/Users/cespenar/sdb/sdb_grid_cpm.db'
    grid_dir = '/Volumes/T3_2TB/sdb/grid_sdb'
    g = SdbGrid(database, grid_dir)

    df = g.df_from_errorbox(star=target)
    chi2_star(star=target, df=df)
    print(df.chi2_star)
