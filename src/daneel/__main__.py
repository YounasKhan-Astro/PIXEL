import datetime
import argparse

from daneel.parameters import Parameters
from daneel.detection.transit import transit as transit_cli


def main():
    parser = argparse.ArgumentParser()

    # -----------------------------
    # Required: input YAML file
    # -----------------------------
    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        required=True,
        help="Input parameter file",
    )

    # -----------------------------
    # Transit flag
    # -----------------------------
    parser.add_argument(
        "-t",
        "--transit",
        dest="transit",
        action="store_true",
        help="Plots transit lightcurve from YAML file",
    )

    # -----------------------------
    # Detection method (Task G)
    # e.g. -d rf
    # -----------------------------
    parser.add_argument(
        "-d",
        "--detect",
        dest="detect",
        type=str,  # <--- IMPORTANT: this takes a string (rf, cnn, ...)
        help="Detection method: rf (Random Forest)",
    )

    # -----------------------------
    # Atmosphere (unused placeholder)
    # -----------------------------
    parser.add_argument(
        "-a",
        "--atmosphere",
        dest="atmosphere",
        action="store_true",
        help="Atmospheric characterisation",
    )

    # Parse arguments
    args = parser.parse_args()

    # Start log
    start = datetime.datetime.now()
    print(f"Daneel starts at {start}")

    # ===========================
    # Transit mode
    # ===========================
    if args.transit:
        transit_cli(params_yaml=args.input_file)

    # ===========================
    # Detection mode: Random Forest (Task G)
    # ===========================
    elif args.detect == "rf":
        from daneel.detection.rf_detector import run_rf_from_yaml
        run_rf_from_yaml(args.input_file)

    # ===========================
    # Atmosphere mode (not used)
    # ===========================
    elif args.atmosphere:
        input_pars = Parameters(args.input_file).params
        print("Atmosphere mode not implemented yet.")

    # End log
    finish = datetime.datetime.now()
    print(f"Daneel finishes at {finish}")


if __name__ == "__main__":
    main()
