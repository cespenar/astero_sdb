from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pandas.core.frame import DataFrame, Series

from .gyre_reader import GyreData
from .sdb_grid_reader import SdbGrid
from .star import Star

plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.labelspacing'] = 0.1
plt.rcParams['errorbar.capsize'] = 2


def plot_hr_logg_teff(targets: list[Star],
                      star_name: str,
                      df: DataFrame,
                      column: str,
                      out_folder: Path,
                      number_of_models: int,
                      sigma_range: int = 3,
                      threshold_chi2: float = None,
                      threshold_chi2_mp: float = None,
                      label_x: float = None,
                      label_y: float = None,
                      x_lim: tuple = None,
                      y_lim: tuple = None) -> None:
    plt.figure()
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xlabel(r'$T_\mathrm{eff}\/\mathrm{[\times 10^3\,K]}$')
    plt.ylabel(r'$\log\,g$')

    styles_observed = ['mD', 'gs', ]
    error_bar_colors = ['magenta', 'green', ]

    chi2_min = df[f'{column}'].min()

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
        _plot_target_logg_teff(target=target, style=style, error_color=color,
                               marker_size=2.0, plot_error=False)
        for sigma in range(1, sigma_range + 1):
            _plot_error_box_logg_teff(star=target, color=color, sigma=sigma)

    col_nr = column.split('_')[-1]
    plt.text(label_x, label_y,
             f'{star_name}\n'
             fr'$\chi^2$'
             f', set {col_nr}'
             )

    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)

    plt.legend(loc='upper left')

    output = out_folder.joinpath(f'{star_name}_logg-Teff-best_{column}')
    plt.savefig(output.with_suffix('.pdf'), format='pdf', bbox_inches='tight')
    plt.close()


def _plot_target_logg_teff(target: Star,
                           style: str,
                           error_color: str,
                           marker_size: float = 5,
                           plot_error: bool = False) -> None:
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
                   number_of_models: int,
                   threshold_chi2: float = None,
                   threshold_chi2_mp: float = None) -> None:
    chi2_min = df[f'{column}'].min()

    if threshold_chi2:
        df = df[df[f'{column}'] <= threshold_chi2]
    if threshold_chi2_mp:
        df = df[df[f'{column}'] <= threshold_chi2_mp * chi2_min]

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
            f'{"age":>5}',
            f'{"m_sdb":>6}',
            f'{"r_sdb":>6}',
            f'{"chi2_old":>8}',
            f'{"chi2_new":>8}',
            '\n',
        ])
        f.write(header)

        for _, model in df[['id', f'{column}', 'z_i', 'm_i', 'm_env', 'y_i',
                            'he4', 'log_Teff', 'log_L', 'log_g', 'age', 'm',
                            'radius', 'chi2_star_1',
                            'chi2_star_2']].sort_values(f'{column}').head(
            number_of_models).iterrows():
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
                f'{model.age / 1e9:>5.3f}',
                f'{model.m:>6.4f}',
                f'{model.radius:>6.4f}',
                f'{model.chi2_star_1:>8.2f}',
                f'{model.chi2_star_2:>8.2f}',
                '\n'
            ])
            f.write(row)


def plot_modes(star: Star,
               df: DataFrame,
               grid: SdbGrid,
               grid_dest_dir: str,
               column: str,
               out_folder: Path,
               number_of_models: int,
               x_lim: tuple = (0, 10000.0),
               star_name: str = None) -> None:
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
                                 x_lim=x_lim)


def _plot_modes_single_model(star: Star,
                             puls_data: GyreData,
                             model: Series,
                             col: str,
                             output: Path,
                             x_lim: tuple,
                             plot_legend: bool = False,
                             legend_loc: str = None) -> None:
    w, h = plt.figaspect(0.3)
    plt.figure(figsize=(w, h))
    plt.xlabel(r'$P\/\mathrm{{[s]}}$')
    plt.ylabel('')

    plt.vlines(x=puls_data.periods(deg=1, g_modes_only=True),
               ymin=0, ymax=1, colors='black', linewidth=2)
    plt.vlines(x=star.frequencies['P'], ymin=0, ymax=1, colors='red',
               linestyles='dotted', linewidth=4)

    plt.text(200, 1.05,
             f'{star.name}'
             fr'$,\/\chi^2={model[f"{col}"]:.2f},\/$'
             fr'$\mathrm{{id}}={model.id},\/Z_\mathrm{{i}}={model.z_i:.3f},\/$'
             fr'$Y_\mathrm{{i}}={model.y_i:.4f},\/M_\mathrm{{i}}={model.m_i:.2f}\,M_\odot,\/$'
             fr'$M_\mathrm{{env}}={model.m_env:.4f}\,M_\odot,\/Y_\mathrm{{c}}={model.he4:.2f},\/$'
             fr'$T_\mathrm{{eff}}={10 ** model.log_Teff:.0f}\,\mathrm{{K}},\/\log\,g={model.log_g:.3f}$'
             )

    plt.xlim(x_lim)
    plt.ylim(-0.02, 1.15)

    if plot_legend and legend_loc:
        plt.legend(loc=legend_loc)

    plt.savefig(output.with_suffix(output.suffix + '.pdf'), format='pdf',
                bbox_inches='tight')
    plt.close()