# eummy: A tool to create color images from Euclid MER stacks

**eummy** is a Python tool designed to create high-quality color images from Euclid space telescope data, more specifically the **Euclid MER** stacked FITS images per tile.

For more details about the Euclid space telescope visit <https://euclid-ec.org>.

## Installation

You can install **eummy** directly from PyPI:

```bash
pip install eummy
```
Note: This will automatically install required dependencies including numpy, astropy, opencv-python, tifffile, scikit-image, scipy, and numexpr.

## Usage

**eummy** is designed to be used as a command-line tool. After installation, the eummy command will be available in your terminal.

If your FITS files follow the standard Euclid MER naming convention, simply provide the path to the directory containing the 4 stacked images (1 VIS, 3 NISP):

```bash
eummy --path /path/to/MERstacks/
```

This is sufficient to create a good color image for many purposes. To fine-tune parameters, invoke

```bash
eummy --help
```
for options.
