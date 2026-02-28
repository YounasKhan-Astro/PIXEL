import datetime
import argparse
from .dream.gan_dream import GANDreamer  # for Assignment2 Task I
from daneel.parameters import Parameters
from daneel.detection.transit import transit as transit_cli


def main():

    parser = argparse.ArgumentParser()

    # Required YAML parameter file
    parser.add_argument(
        "-i", "--input",
        dest="input_file",
        type=str,
        required=True,
        help="Input parameter file",
    )
    # Transit mode
    parser.add_argument(
        "-t", "--transit",
        dest="transit",
        action="store_true",
        help="Plots transit lightcurve from YAML file",
    )

    # Detection mode (RF or CNN)
    parser.add_argument(
        "-d", "--detect",
        dest="detect",
        type=str,
        help="Detection method: rf or cnn",
    )

    # Atmosphere (unused)
    parser.add_argument(
        "-a", "--atmosphere",
        dest="atmosphere",
        action="store_true",
        help="Atmospheric characterisation",
    )
    #assignment2 Task I
    parser.add_argument(
    "--dream",
    action="store_true",
    help="Train a GAN to dream new exoplanetary transit light curves",
)

    args = parser.parse_args()
    #assignemnt 2 Task I
    parameters = Parameters(args.input_file)
    params = parameters.params

    # Start log
    start = datetime.datetime.now()
    print(f"Daneel starts at {start}")

    # ===========================
    # Transit mode
    # ===========================
    if args.transit:
        transit_cli(params_yaml=args.input_file)
    # Assignent 2 Task I
    if args.dream:
        dreamer = GANDreamer(params)
        dreamer.run()
        return
    # ===========================
    # Random Forest mode
    # ===========================
    elif args.detect == "rf":
        from daneel.detection.rf_detector import run_rf_from_yaml
        run_rf_from_yaml(args.input_file)

    # ===========================
    # CNN mode
    # ===========================
    elif args.detect == "cnn":
        from daneel.detection.cnn_detector import run_cnn_from_yaml
        run_cnn_from_yaml(args.input_file)

    # ===========================
    # Atmosphere mode (unused)
    # ===========================
    elif args.atmosphere:
        print("Atmosphere mode not implemented.")

    else:
        print("No valid mode selected. Use -t, -d rf, -d cnn, or -a.")

    # End log
    finish = datetime.datetime.now()
    print(f"Daneel finishes at {finish}")


if __name__ == "__main__":
    main()
