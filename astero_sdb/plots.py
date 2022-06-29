from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from pandas.core.frame import DataFrame, Series

from .gyre_reader import GyreData
from .sdb_grid_reader import SdbGrid
from .star import Star
from .utils import mass_total_core, mass_conv_core_history, zaehb_age

plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.labelspacing'] = 0.1
plt.rcParams['errorbar.capsize'] = 2


def plot_hr_logg_teff(targets: list[Star],
                      star_name: str,
                      df: DataFrame,
                      column: str,
                      out_folder: Path,
                      number_of_models: int = 50,
                      sigma_range: int = 3,
                      threshold_chi2: float = None,
                      threshold_chi2_mp: float = None,
                      print_name: bool = True,
                      label_x: float = None,
                      label_y: float = None,
                      x_lim: tuple = None,
                      y_lim: tuple = None,
                      save_pdf: bool = True,
                      save_eps: bool = False) -> None:
    """
    Plots logg vs. Teff diagram.

    Parameters
    ----------
    targets : list[Star]
        List of target stars.
    star_name : str
        Name of the star printed in the plot.
    df : DataFrame
        Pandas DataFrame containing the constrained grid.
    column : str
        Name of a column containing S^2 data used to select the best models.
    out_folder : Path
        Output directory.
    number_of_models : int, optional
        Maximum number of fitted models considered for plotting using the
        color scale. Default: 50.
    sigma_range : int, optional
        Maximum value of sigma up to which the error rectangles are plotted.
        Default: 3.
    threshold_chi2 : float | None, optional
        If specified, the maximum value of S^2 considered for plotting using
        the color scale. Default: None.
    threshold_chi2_mp : float | None, optional
        If specified, the maximum value of S^2, expressed as a multiplier of
        S^2_min, considered for plotting using the color scale. Default: None.
    print_name : bool, optional
        If True, prints the name of the star in the plot. Default: True.
    label_x : float | None, optional
        If specified, the x-coordinate (Teff) of the label containg the name
        of the star. Default: None.
    label_y : float | None, optional
        If specified, the y-coordinate (logg) of the label containg the name of
        the star. Default: None.
    x_lim : tuple | None, optional
        If specified, the range of the x-axis (Teff). If None, the range is
        determined automatically. Default: None.
    y_lim : tuple | None, optional
        If specified, the range of the y-axis (logg). If None, the range is
        determined automatically. Default: None.
    save_pdf : bool, optional
        If True, saves a plot as a pdf file. Default: True.
    save_eps : bool, optional
        If True, saves a plot as an eps file. Please note that the plot uses
         transparency, which is not supported by the eps standard.
         Default: False.

    Returns
    -------

    """

    plt.figure()
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xlabel(r'$T_\mathrm{eff}\/\mathrm{[\times 10^3\,K]}$')
    plt.ylabel(r'$\log\,g$')

    styles_observed = ['mD', 'gs', ]
    error_bar_colors = ['magenta', 'green', ]

    chi2_min = df[f'{column}'].min()

    if threshold_chi2_mp:
        number_of_models = len(
            df[df[f'{column}'] <= threshold_chi2_mp * chi2_min])

    plt.scatter(10.0 ** df.sort_values(f'{column}').log_Teff[
                        number_of_models:] / 1000.0,
                df.sort_values(f'{column}').log_g[number_of_models:],
                s=0.2,
                c='black',
                alpha=0.1,
                label=fr'$\mathrm{{sdB\/grid}}$')

    if threshold_chi2:
        df = df[df[f'{column}'] <= threshold_chi2]
    if threshold_chi2_mp:
        df = df[df[f'{column}'] <= threshold_chi2_mp * chi2_min]

    plt.scatter(10.0 ** df.sort_values(f'{column}').log_Teff[
                        :number_of_models] / 1000.0,
                df.sort_values(f'{column}').log_g[:number_of_models],
                s=[20.0 * chi2_min / s for s in df.sort_values(
                    f'{column}')[f'{column}'][:number_of_models]],
                c=df.sort_values(f'{column}')[f'{column}'][:number_of_models],
                cmap='jet_r',
                label=fr'$\mathrm{{best\/models}}$')

    plt.colorbar()

    for target, style, color in zip(targets, styles_observed,
                                    error_bar_colors):
        if target.t_eff and target.log_g:
            _plot_target_logg_teff(target=target, style=style,
                                   error_color=color, marker_size=2.0,
                                   plot_error=False)
            for sigma in range(1, sigma_range + 1):
                _plot_error_box_logg_teff(star=target, color=color,
                                          sigma=sigma)

    if print_name:
        plt.text(label_x, label_y, f'{star_name}\n')

    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)

    plt.legend(loc='upper left')

    output = out_folder.joinpath(f'{star_name}_logg-Teff-best_{column}')
    if save_pdf:
        plt.savefig(output.with_suffix('.pdf'), format='pdf',
                    bbox_inches='tight')
    if save_eps:
        plt.savefig(output.with_suffix('.eps'), format='eps',
                    bbox_inches='tight')
    plt.close()


def _plot_target_logg_teff(target: Star,
                           style: str,
                           error_color: str,
                           marker_size: float = 5,
                           plot_error: bool = False) -> None:
    """
    Plots a single target star in the logg vs. Teff plot.

    Parameters
    ----------
    target : Star
        Target star.
    style : str
        Marker style.
    error_color : str
        Color of the error bar.
    marker_size : float, optional
        Size of the marker. Default: 5.
    plot_error : bool, optional
        If True, plots the error bar of the target. Default: False.

    Returns
    -------

    """
    plt.plot(target.t_eff / 1000.0, target.log_g,
             style, ms=marker_size, label=target.name)
    if plot_error:
        plt.errorbar(target.t_eff / 1000.0, target.log_g,
                     xerr=target.t_eff_err_p / 1000.0, yerr=target.log_g_err_p,
                     fmt='none', ecolor=error_color,
                     elinewidth=1)


def _plot_error_box_logg_teff(star: Star,
                              color: str,
                              sigma: int = 1) -> None:
    """

    Parameters
    ----------
    A helper function to plot error rectangles of the target star in the
    logg vs. Teff plot.

    star : Star
        Target star.
    color : str
        Color of the error rectangles.
    sigma : int, optional
        Value of sigma for which the error rectangle is plotted. Default: 1.

    Returns
    -------

    """

    box = Rectangle(((star.t_eff - sigma * star.t_eff_err_m) / 1000.0,
                     star.log_g - sigma * star.log_g_err_m),
                    sigma * (star.t_eff_err_p + star.t_eff_err_m) / 1000.0,
                    sigma * (star.log_g_err_p + star.log_g_err_m),
                    linewidth=0.5,
                    edgecolor=color,
                    facecolor='none')
    ax = plt.gca()
    ax.add_patch(box)


def save_best_info(star_name: str,
                   df: DataFrame,
                   column: str,
                   out_folder: Path,
                   grid: SdbGrid,
                   number_of_models: int = None,
                   threshold_chi2: float = None,
                   threshold_chi2_mp: float = None,
                   calculate_age_sdb: bool = False,
                   calculate_m_core: bool = False) -> None:
    """
    Saves a file containing parameters of the best models, based on the
    selected number of best models or S^2 thresholds.

    Parameters
    ----------
    star_name : str
        Name of the star printed in the plot.
    df : DataFrame
        Pandas DataFrame containing the constrained grid.
    column : str
        Name of a column containing S^2 data used to select the best models.
    out_folder : Path
        Output directory.
    grid : SdbGrid
        SdbGrid object containing the grid.
    number_of_models : int | None, optional
        If specified, maximum number of models for which the parameters are
        saved. If None, all models within the specified S^2 threshold are
         taken into account. Default: None.
    threshold_chi2 : float | None, optional
        If specified, the maximum value of S^2 considered for selecting the
        models. Default: None.
    threshold_chi2_mp : float | None, optional
        If specified, the maximum value of S^2, expressed as a multiplier of
        S^2_min, considered for selecting the models. Default: None.
    calculate_age_sdb : bool, optional
        If True, calculates and saves the age on the extreme horizontal branch.
        Default: False.
    calculate_m_core : bool, optional
        If True, calculates and saves the mass of the core, defined as a sum of
        the mass of the convective core and the mass of the natural
        semiconvective zone (c.f. Ostrowski et al. 2021). Default: False.

    Returns
    -------

    """

    chi2_min = df[f'{column}'].min()

    if threshold_chi2:
        df = df[df[f'{column}'] <= threshold_chi2]
        if not number_of_models:
            number_of_models = 50
    if threshold_chi2_mp:
        df = df[df[f'{column}'] <= threshold_chi2_mp * chi2_min]
        if not number_of_models:
            number_of_models = len(df)

    output = out_folder.joinpath(f'{star_name}-best-{column}.txt')
    with output.open(mode='w') as f:
        header = ' '.join([
            f'{"id":>6}',
            f'{"chi2":>9}',
            f'{"z_i":>5}',
            f'{"m_i":>4}',
            f'{"m_env":>6}',
            f'{"y_i":>6}',
            f'{"he4":>4}',
            f'{"T_eff":>5}',
            f'{"log_L":>5}',
            f'{"log_g":>5}',
            f'{"age":>6}',
            f'{"m_sdb":>6}',
            f'{"r_sdb":>6}',
            f'{"age_sdb":>8}',
            f'{"m_cc":>6}',
            f'{"m_tc":>6}',
            '\n',
        ])
        f.write(header)

        for _, model in df[['id', f'{column}', 'z_i', 'm_i', 'm_env', 'y_i',
                            'he4', 'log_Teff', 'log_L', 'log_g', 'age', 'm',
                            'radius', 'model_number', 'top_dir',
                            'log_dir']].sort_values(f'{column}').head(
            number_of_models).iterrows():
            if calculate_age_sdb:
                history = grid.read_history(log_dir=model.log_dir,
                                            top_dir=model.top_dir,
                                            dest_dir=grid.grid_dir,
                                            delete_file=False,
                                            keep_tree=True)
                age_sdb = (model.age - zaehb_age(history_data=history)) / 1e6
                if age_sdb < 0:
                    age_sdb = -1.0
            else:
                age_sdb = -1.0

            if calculate_m_core:
                history = grid.read_history(log_dir=model.log_dir,
                                            top_dir=model.top_dir,
                                            dest_dir=grid.grid_dir,
                                            delete_file=False,
                                            keep_tree=True)
                profile = grid.read_evol_model(log_dir=model.log_dir,
                                               top_dir=model.top_dir,
                                               he4=round(model.he4, 2),
                                               dest_dir=grid.grid_dir,
                                               delete_file=False,
                                               keep_tree=True)
                m_cc = mass_conv_core_history(history=history,
                                              model_nr=model.model_number)
                m_tc = mass_total_core(profile=profile)
            else:
                m_cc = -1.0
                m_tc = -1.0

            row = ' '.join([
                f'{model.id:>6.0f}',
                f'{model[f"{column}"]:>9.2f}',
                f'{model.z_i:>5.3f}',
                f'{model.m_i:>4.2f}',
                f'{model.m_env:>6.4f}',
                f'{model.y_i:>6.4f}',
                f'{model.he4:>4.2f}',
                f'{10 ** model.log_Teff:>5.0f}',
                f'{model.log_L:>5.3f}',
                f'{model.log_g:>5.3f}',
                f'{model.age / 1e9:>6.3f}',
                f'{model.m:>6.4f}',
                f'{model.radius:>6.4f}',
                f'{age_sdb:>8.2f}',
                f'{m_cc:>6.4f}',
                f'{m_tc:>6.4f}',
                '\n'
            ])
            f.write(row)


def plot_modes(star: Star,
               df: DataFrame,
               grid: SdbGrid,
               grid_dest_dir: Path,
               column: str,
               out_folder: Path,
               number_of_models: int = 50,
               threshold_chi2_mp: float = None,
               x_lim: tuple = (0, 10000.0),
               star_name: str = None,
               save_pdf: bool = True,
               save_eps: bool = False) -> None:
    """
    Plots selected observed periods vs. theoretical periods from GYRE models
    for all models selected by specifying the number of best models
    or S^2 threshold.

    Parameters
    ----------
    star : Star
        Target star.
    df : DataFrame
        Pandas DataFrame containing the constrained grid.
    grid : SdbGrid
        SdbGrid object containing the grid.
    grid_dest_dir : Path
        Temporary directory for the extracted pulsation model.
    column : str
        Name of a column containing S^2 data used to select the best models.
    out_folder : Path
        Output directory.
    number_of_models : int, optional
        Maximum number of models for which the periods are plotted.
        Default: 50.
    threshold_chi2_mp : float | None, optional
        If specified, the maximum value of S^2, expressed as a multiplier of
        S^2_min, considered for selecting the models for which the periods
        are plotted. Default: None.
    x_lim : tuple, optional
        The range of the x-axis (P) in seconds. Default: (0, 10000.0).
    star_name : str | None, optional
        If specified, the name of the star used as a label in the plot. If None
        automatically derived from the `star` parameter. Default: None.
    save_pdf : bool, optional
        If True, saves a plot as a pdf file. Default: True.
    save_eps : bool, optional
        If True, saves a plot as an eps file. Default: False.

    Returns
    -------

    """

    chi2_min = df[f'{column}'].min()
    if threshold_chi2_mp:
        number_of_models = len(
            df[df[f'{column}'] <= threshold_chi2_mp * chi2_min])

    for _, model in df.sort_values(f'{column}').head(
            number_of_models).iterrows():
        puls_data = grid.read_puls_model(
            log_dir=model.log_dir,
            top_dir=model.top_dir,
            he4=model.he4,
            dest_dir=grid_dest_dir,
            delete_file=False,
            keep_tree=True
        )
        if not star_name:
            star_name = star.name
        output = out_folder.joinpath(
            f'{star_name}-modes-{column}-chi2_{model[f"{column}"]:.2f}'
            f'-id_{model.id}')
        _plot_modes_single_model(star=star,
                                 puls_data=puls_data,
                                 model=model,
                                 col=column,
                                 output=output,
                                 x_lim=x_lim,
                                 star_name=star_name,
                                 save_pdf=save_pdf,
                                 save_eps=save_eps)


def _plot_modes_single_model(star: Star,
                             puls_data: GyreData,
                             model: Series,
                             col: str,
                             output: Path,
                             x_lim: tuple,
                             star_name: str,
                             plot_legend: bool = False,
                             legend_loc: str = None,
                             save_pdf: bool = True,
                             save_eps: bool = False) -> None:
    """
    Plots selected observed periods vs. theoretical periods for a single GYRE
    model.

    Parameters
    ----------
    star : Star
        Target star.
    puls_data : GyreData
        Pulsation model as a GyreData object.
    model : Series
        Data for a single evolutionary model corresponding to the considered
        pulsation model.
    col : str
        Name of a column containing S^2 data used to select the best models.
    output : Path
        Path of the output file.
    x_lim : tuple, optional
        The range of the x-axis (P) in seconds.
    star_name : str
        The name of the star
    plot_legend : bool, optional
        If True, plots the legend. Default: False.
    legend_loc : str | None, optional
        If specified, location of the legend in the descriptive Matplotlib
        format. Default: None.
    save_pdf : bool, optional
        If True, saves a plot as a pdf file. Default: True.
    save_eps : bool, optional
        If True, saves a plot as an eps file. Default: False.

    Returns
    -------

    """

    periods = star.periods_explicit()[0]
    degrees = np.sort(np.unique([p['l'] for p in periods.values()]))

    colors = ['red', 'green', 'blue', 'magenta']

    w, h = plt.figaspect(0.3 * len(degrees))
    fig = plt.figure(figsize=(w, h))
    gs = fig.add_gridspec(len(degrees), hspace=0)
    axs = gs.subplots(sharex=True, squeeze=False)

    for i, deg in enumerate(degrees):
        periods_deg = [p['P'] for p in periods.values() if p['l'] == deg]
        if deg == degrees[-1]:
            axs[i, 0].set_xlabel(r'$P\/\mathrm{{[s]}}$')
        axs[i, 0].set_ylabel(fr'$\ell={int(deg)}$')

        axs[i, 0].set_ylim(0.0, 1.0)

        axs[i, 0].tick_params(axis='y',
                              which='both',
                              left=False,
                              right=False,
                              labelleft=False,
                              labelright=False)

        axs[i, 0].vlines(x=puls_data.periods(deg=deg, g_modes_only=True),
                         ymin=0, ymax=1, colors='black', linewidth=1)
        axs[i, 0].vlines(x=periods_deg, ymin=0, ymax=1, colors=colors[i],
                         linestyles='dotted', linewidth=4)

        if i == 0:
            axs[i, 0].text(x_lim[0] + int(0.02 * (x_lim[1] - x_lim[0])), 1.1,
                           f'{star_name}'
                           fr'$,\/\chi^2={model[f"{col}"]:.2f},\/$'
                           fr'$\mathrm{{id}}={model.id},\/Z_\mathrm{{i}}={model.z_i:.3f},\/$'
                           fr'$Y_\mathrm{{i}}={model.y_i:.4f},\/M_\mathrm{{i}}={model.m_i:.2f}\,M_\odot,\/$'
                           '\n'
                           fr'$M_\mathrm{{env}}={model.m_env:.4f}\,M_\odot,\/Y_\mathrm{{c}}={model.he4:.2f},\/$'
                           fr'$T_\mathrm{{eff}}={10 ** model.log_Teff:.0f}\,\mathrm{{K}},\/\log\,g={model.log_g:.3f}$'
                           )

    plt.xlim(x_lim)

    if plot_legend and legend_loc:
        plt.legend(loc=legend_loc)

    if save_pdf:
        plt.savefig(output.with_suffix(output.suffix + '.pdf'), format='pdf',
                    bbox_inches='tight')
    if save_eps:
        plt.savefig(output.with_suffix(output.suffix + '.eps'), format='eps',
                    bbox_inches='tight')
    plt.close()
