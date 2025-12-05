# Daneel

A practical example to detect and characterize exoplanets.

The full documentation is at https://tiziano1590.github.io/comp_astro_25/index.html

## Installation

### Prerequisites

- Python >= 3.10

### Install from source

```bash
git clone https://github.com/tiziano1590/comp_astro_25.git
cd comp_astro_25
pip install .
```

### Development installation

```bash
git clone https://github.com/tiziano1590/comp_astro_25.git
cd comp_astro_25
pip install -e .
```

## Usage

After installation, you can run daneel from the command line:

```bash
daneel -i <input_file> [options]
```

### Command-line options

- `-i, --input`: Input parameter file (required)
- `-d, --detect`: Initialize detection algorithms for exoplanets
- `-a, --atmosphere`: Atmospheric characterization from input transmission spectrum

### Examples

```bash
# Run exoplanet detection
daneel -i parameters.yaml -d

# Run atmospheric characterization
daneel -i parameters.yaml -a

# Run both detection and atmospheric analysis
daneel -i parameters.yaml -d -a
```

## Input File Format

The input file should be a YAML file containing the necessary parameters for the analysis.

## License

This project is licensed under the MIT License.

## Author

Tiziano Zingales (tiziano.zingales@unipd.it)

## Transit Light Curve (Task F & G)

 Daneel can compute and plot a planetary transit light curve using the `-t` / `--transit` flag.
 The transit parameters must be provided inside the input YAML file under the key `transit:`.

## Example `params.yaml` snippet

```yaml
transit:
  t0: 0.0
  per: 74.92768653
  rp: 0.0643
  a: 79.51
  inc: 89.47
  ecc: 0.0
  w: 90.0
  u: [0.3, 0.2]
  limb_dark: quadratic
  ```
## Run the transit model

To generate the light curve directly from the command line:
 ```bash
 daneel -i examples/params.yaml -t
 ```
 This command:

loads the transit parameters from the YAML file

builds the transit model using the batman package

plots the transit light curve

saves the output figure to lc.png
## Output

Running the command will produce a PNG file containing the transit light curve:
```lc.png
```
To run Random Forest detection:

    daneel -i parameters.yaml -d rf
## Random Forest detection (Assignment 2 – Task G)

Daneel includes a Random Forest–based exoplanet detector implemented in
`src/daneel/detection/rf_detector.py`. The configuration is read from the
YAML parameter file in the `detection` section.

Example `parameters.yaml` snippet:

```yaml
detection:
  method: rf                 
  kernel: default_rf         
  csv_path: tess_data.csv    
  n_bins: 1000
  use_scaler: false
  samples_per_class: 350
  n_estimators: 500
  max_depth: null
