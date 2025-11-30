import numpy as np
import batman
import matplotlib.pyplot as plt
import yaml 

class TransitModel:
    """
    Transit model class for exoplanet detection using batman.

    This class generates synthetic light curves for exoplanet transits
    and provides visualization capabilities.
    """

    def __init__(self, params_dict):
        """
        Initialize the transit model with parameters from configuration file.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing transit parameters
        """
        # Batman transit parameters object
        self.params = batman.TransitParams()

        # Basic orbital / geometry parameters
        self.params.t0 = params_dict.get("t0", 0.0)       # mid-transit time [days]
        self.params.per = params_dict.get("per")          # orbital period [days]
        self.params.rp = params_dict.get("rp")            # Rp/R*
        self.params.a = params_dict.get("a")              # a/R*
        self.params.inc = params_dict.get("inc")          # inclination [deg]
        self.params.ecc = params_dict.get("ecc", 0.0)     # eccentricity
        self.params.w = params_dict.get("w", 90.0)        # arg of periastron [deg]

        # Limb darkening
        self.params.u = params_dict.get("u", [0.357, 0.225])  # limb darkening coeffs for Kepler 297 star
        self.params.limb_dark = params_dict.get(
            "limb_dark", "quadratic"
        )  # limb darkening law

        # Time array around mid-transit (in days)
        # Here we look at +/- 0.75 days around t0 with 2000 points
        self.t = np.linspace(-0.75, 0.75, 2000)

        # Initialize batman model
        self.model = batman.TransitModel(self.params, self.t)

    def compute_light_curve(self):
        """
        Compute the light curve using the batman transit model.

        Returns
        -------
        flux : ndarray
            Array of relative flux values
        """
        self.flux = self.model.light_curve(self.params)
        return self.flux

    def plot_light_curve(self, output_file="lc.png"):
        """
        Plot and save the transit light curve.

        Parameters
        ----------
        output_file : str, optional
            Output filename for the plot (default: 'lc.png')
        """
        if not hasattr(self, "flux"):
            self.compute_light_curve()

        plt.figure(figsize=(10, 6))
        plt.plot(self.t, self.flux)
        plt.xlabel("Time from central transit (days)")
        plt.ylabel("Relative flux")
        plt.title("Transit light curve for Kepler-297 c")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.show()
        print(f"Light curve saved to {output_file}")

    def run(self, output_file="lc.png"):
        """
        Run the complete transit model workflow: compute and plot light curve.
        """
        self.compute_light_curve()
        self.plot_light_curve(output_file)


def kepler_297c_params():
    """
    Parameters for Kepler-297 c, based on exoplanet.eu values.
    This is what you used for Task F.
    """
    # Given values from the exoplanet catalogue
    P_days = 74.92768653
    a_AU = 0.3292
    Rp_Rj = 0.57
    R_star_Rsun = 0.89

    # Conversion factors
    RJ_OVER_RSUN = 0.10045   # 1 R_J in units of R_sun
    AU_OVER_RSUN = 215.032   # 1 AU in units of R_sun

    # Dimensionless ratios needed by batman
    rp_over_rstar = Rp_Rj * RJ_OVER_RSUN / R_star_Rsun
    a_over_rstar = a_AU * AU_OVER_RSUN / R_star_Rsun

    params_dict = {
        "t0": 0.0,
        "per": P_days,
        "rp": rp_over_rstar,
        "a": a_over_rstar,
        "inc": 89.47,
        "ecc": 0.0,
        "w": 90.0,
        "u": [0.3, 0.2],
        "limb_dark": "quadratic",
    }

    return params_dict


def transit_from_yaml(yaml_path, output_file="transit_cli.png"):
    """
    Run the transit model using parameters loaded from a YAML file.

    This is the function that will be called by the CLI (Task G).
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # If YAML has a 'transit' block, use it, otherwise use the whole file
    if "transit" in config:
        params_dict = config["transit"]
    else:
        params_dict = config

    model = TransitModel(params_dict)
    model.run(output_file=output_file)
def transit(params_yaml=None, out_png=None):
    """
    Function required by the CLI for Task G/H.
    This is what Daneel calls when you run:
        daneel -i params.yaml -t
    """
    if params_yaml is None:
        # Default: use your Task F planet Kepler-297c
        params = kepler_297c_params()
        output_file = out_png or "assignment2_taskA.png"
        model = TransitModel(params)
        model.run(output_file=output_file)
    else:
        # Use parameters from YAML file (Task G/H)
        output_file = out_png or "lc.png"
        transit_from_yaml(params_yaml, output_file=output_file)


if __name__ == "__main__":
    # Build parameter dictionary for Kepler-297 c
    params = kepler_297c_params()

    # Create model and run it, saving to the required filename
    model = TransitModel(params)
    model.run(output_file="assignment2_taskA.png")
