# eummy: A tool to create color images from Euclid MER stacks

**eummy** is a Python tool designed to create high-quality color images from **Euclid MER** stacked images per tile. It processes FITS images from VIS and NISP instruments to produce visually optimized color composites.

## Installation

You can install Eummy directly from PyPI:

```bash
pip install eummy
```
Note: This will automatically install required dependencies including numpy, astropy, opencv-python, tifffile, scikit-image, scipy, and numexpr.

## Usage

**eummy** is designed to be used as a command-line tool. After installation, the eummy command will be available in your terminal.

### Basic Usage

If your FITS files follow the standard Euclid MER naming convention, simply provide the path to the directory containing the 4 stacked images (1 VIS, 3 NISP):

```bash
eummy --path /path/to/MERstacks/
```

This is sufficient to create a good color image for many purposes. To fine-tune parameters, invoke

```bash
eummy --help
```
for options.

