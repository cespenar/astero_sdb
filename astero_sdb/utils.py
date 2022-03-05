import numpy as np
from mesa_reader import MesaData


def zaehb_index(history_data: MesaData) -> int:
    """Finds index of the ZAEHB model.

    Parameters
    ----------
    history_data : mesa_reader.MesaData
        Evolutionary track (MESA history file) as MesaData object.

    Returns
    ----------
    int
        Index of the ZAEHB model.

    """

    return history_data.model_number[history_data.mass_conv_core > 0.0][0] - 1


def zaehb_age(history_data: MesaData,
              zaehb_index: int) -> float:
    """Finds age of the ZAEHB model calculated from the beginning of the PMS
    evolution.

    Parameters
    ----------
    history_data : mesa_reader.MesaData
        Evolutionary track (MESA history file) as MesaData object.

    Returns
    ----------
    int
        Age of the ZAEHB model.

    """

    return history_data.star_age[zaehb_index]


def find_semiconvection_bottom(profile: MesaData,
                               verbose: bool = False) -> int:
    """Finds the zone where the natural semiconvection starts in a MESA model.

    Parameters
    ----------
    profile : mesa_reader.MesaData
        Evolutionary model (MESA profile file) as MesaData object.

    Returns
    ----------
    zone_semi_bottom : int
        The zone where the natural semiconvection starts. If negative, the
        semiconvective zone was not found.

    """

    zone_semi_bottom = -1
    delta_nabla = (profile.gradr - profile.gradL)[::-1]
    for i, delta in enumerate(delta_nabla):
        if i == 0:
            if delta <= 0.0:
                break
            else:
                continue
        if delta_nabla[i - 1] * delta <= 0.0:
            zone_semi_bottom = profile.zone[::-1][i - 1]
            break
    if verbose:
        print('zone_semi_bottom =', zone_semi_bottom)
    return zone_semi_bottom


def find_semiconvection_top(profile: MesaData,
                            zone_semi_bottom: int,
                            verbose: bool = False) -> int:
    """Finds the zone where the natural semiconvection ends in a MESA model.

    Parameters
    ----------
    profile : mesa_reader.MesaData
        Evolutionary model (MESA profile file) as MesaData object.
    zone_semi_bottom : int
        The zone where the natural semiconvection starts.

    Returns
    ----------
    zone_semi_top : int
        The zone where the natural semiconvection ends.

    """

    ind_q08 = np.where(profile.q >= 0.8)[0][-1]
    zone_semi_top = np.argmax(
        profile.gradL[ind_q08:zone_semi_bottom - 1]) + ind_q08

    in_conv = False
    delta_nabla = (profile.gradr - profile.gradL)[
                  zone_semi_top - 1:zone_semi_bottom - 1]
    for i, delta in enumerate(delta_nabla):
        if not in_conv and delta <= 0:
            continue
        elif not in_conv and delta > 0:
            in_conv = True
            if zone_semi_top - 1 + i >= zone_semi_bottom:
                print('zone_semi_top >= zone_semi_bottom; this is wrong!')
                break
            else:
                continue
        elif in_conv and delta <= 0:
            zone_semi_top = zone_semi_top + i - 1
            break
    if verbose:
        print('zone_semi_top =', zone_semi_top)
    return zone_semi_top
