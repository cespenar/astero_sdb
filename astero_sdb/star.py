from copy import deepcopy

import numpy as np


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
        
        periods = [{},]
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
                        id_middle = round((df_multi['id'][0] + df_multi['id'][1])/2.0, 1)
                        p_middle = round((df_multi['P'][0] + df_multi['P'][1])/2.0, 5)
                        p_dict[id_middle] = {'P': p_middle, 'l': l}
                else:
                    periods_temp = []
                    for p_dict in periods:
                        p_dict_temp = deepcopy(p_dict)
                        p_dict[df_multi['id'][0]] = {'P': df_multi['P'][0], 'l': l}
                        p_dict_temp[df_multi['id'][1]] = {'P': df_multi['P'][1], 'l': l}
                        periods_temp.append(p_dict_temp)
                    for p_dict in periods_temp:
                        periods.append(p_dict) 
        return periods
