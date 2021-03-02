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
                self.name, dtype=None, skip_header=1, names=True)
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


if __name__ == "__main__":
    pass
