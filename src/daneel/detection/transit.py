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
        plt.title("Transit light curve for Kepler-297c")
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
        "u":   [0.357, 0.225],
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
        output_file = out_png or "Kepler-297c_assignment1_taskF.png"
        model = TransitModel(params)
        model.run(output_file=output_file)
    else:
        # Use parameters from YAML file (Task G/H)
        output_file = out_png or "lc.png"
        transit_from_yaml(params_yaml, output_file=output_file)
def two_planet_transits_taskB():
    """
    Assignment 2 – Task B:
    Plot two transiting planets around the same star.
    Planet 1: original Kepler-297 c
    Planet 2: same orbit but radius reduced by a factor 1/2.
    The figure is saved as 'assignment2_taskB.png'.
    """
    # Base planet = your Kepler-297 c from Task A
    params1 = kepler_297c_params()

    # Second planet: identical parameters but Rp -> Rp/2
    params2 = params1.copy()
    params2["rp"] = params1["rp"] * 0.5  # radius scaled by 1/2

    # Build models
    model1 = TransitModel(params1)
    model2 = TransitModel(params2)

    flux1 = model1.compute_light_curve()
    flux2 = model2.compute_light_curve()

    t = model1.t  # time array (same for both)

    # Plot both light curves on the same axes
    plt.figure(figsize=(10, 6))
    plt.plot(t, flux1, label="Planet 1: Kepler-297 c", linewidth=2)
    plt.plot(t, flux2, label="Planet 2: Rp = 0.5 × Rp₁", linewidth=2)

    plt.xlabel("Time from central transit (days)")
    plt.ylabel("Relative flux")
    plt.title("Transits of two planets around Kepler-297")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assignment2_taskB.png")
    plt.show()

    print("Figure saved to assignment2_taskB.png")
def three_planet_transits_taskC():
    """
    Assignment 2 – Task C:
    Plot transits of three planets around the same star:
      - Planet 1: Kepler-297 c (original)
      - Planet 2: radius = 0.5 × Rp_c
      - Planet 3: radius = 2.0 × Rp_c

    The figure is saved as 'assignment2_taskC.png'.
    """
    # Base planet: Kepler-297 c
    params_base = kepler_297c_params()

    # Planet 2: half the radius
    params_small = params_base.copy()
    params_small["rp"] = params_base["rp"] * 0.5

    # Planet 3: double the radius
    params_big = params_base.copy()
    params_big["rp"] = params_base["rp"] * 2.0

    # Build models and compute light curves
    model_base = TransitModel(params_base)
    flux_base = model_base.compute_light_curve()

    model_small = TransitModel(params_small)
    flux_small = model_small.compute_light_curve()

    model_big = TransitModel(params_big)
    flux_big = model_big.compute_light_curve()

    # Time grid (same for all)
    t = model_base.t

    # Plot all three with different colours and legend
    plt.figure(figsize=(10, 6))
    plt.plot(t, flux_base, label="Kepler-297 c (Rp)", linewidth=2)
    plt.plot(t, flux_small, label="Planet 2 (0.5 × Rp)", linewidth=2)
    plt.plot(t, flux_big, label="Planet 3 (2 × Rp)", linewidth=2)

    plt.xlabel("Time from central transit (days)")
    plt.ylabel("Relative flux")
    plt.title("Transit light curves – Kepler-297 family (Task C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assignment2_taskC.png")
    plt.show()

    print("Figure saved to assignment2_taskC.png")


if __name__ == "__main__":
    # Build parameter dictionary for Kepler-297 c
    params = kepler_297c_params()

    # Create model and run it, saving to the required filename
    model = TransitModel(params)
    model.run(output_file="assignment2_taskA.png")
