#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# eummy.py - A program to create color images from Euclid MER stacks

# MIT License

# Copyright (c) [2026] [Mischa Schirmer]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""eummy_dev.py -- entry point.

The actual pipeline lives in three modules:
  eummy_io.py          file/header I/O, argument parsing, cutout-spec parsing
  eummy_fluxspace.py    raw bands -> stretched/contrast-adjusted L/B/G/R
  eummy_colorspace.py   Lab construction, RGB reconstruction, cutout render

This file just wires them together.
"""

import cv2
import numexpr as ne

from .eummy_io import parse_arguments, load_cutout_specs
from .eummy_fluxspace import (rescale_and_blend, _apply_cvd_daltonize,
                               stretch_and_normalise_channel)
from .eummy_colorspace import process_cutouts, colorise_image


def main():
    args, parser = parse_arguments()
    ne.set_num_threads(args.nthreads)
    cv2.setNumThreads(args.nthreads)

    # 1. Photometric rescaling, bad-pixel repair, channel blending
    B, G, R, L, wcs, fits_path, nan_mask = rescale_and_blend(args, parser)

    # 2. CVD daltonization — applied before normalisation, on raw flux
    if args.cvd is not None:
        print(f"Applying {args.cvd_type} daltonization (k={args.cvd:.2f})")
        _apply_cvd_daltonize(B, G, R, args.cvd, args.cvd_type)

    # 3. Stretch B/G/R with the same asinh function used for L, before ever
    #    reaching the RGB->XYZ->Lab step (reproduces the historical
    #    eummy_21_/eummy_stable per-channel nonlinear-stretch behaviour).
    #    L stays raw here -- stretch_and_normalise_channel() handles it
    #    separately inside colorise_image / _render_cutout_crop, since it
    #    always needs the dynamic-range stretch applied to become the Lab
    #    lightness channel.
    for ch in (B, G, R):
        stretch_and_normalise_channel(ch, args)

    # Resolve --contrast to a concrete number once here, printing the single
    # announcement for the whole run. contrast_adjustment() itself never
    # prints (see its docstring) -- it runs once for the full image, or once
    # per cutout when cutouts are requested, and by resolving here first
    # (using the full mosaic's shape, not a cutout's small crop) every call
    # downstream just sees an already-concrete value and stays silent.
    if args.contrast is None:
        height, width = L.shape
        if height == 10200 and width == 10200:
            args.contrast = 1.6
            print(f"Enhancing contrast by {args.contrast} (default for EDS)")
        else:
            args.contrast = 1.0
            print(f"Enhancing contrast by {args.contrast} (default for EWS)")
    elif args.contrast != 0:
        print(f"Enhancing contrast by {args.contrast}")

    # Cutouts are extracted here while only the four float32 channel arrays
    # are live.  Each cutout runs the full colour pipeline on its small crop.
    # When cutouts are requested the full mosaic TIFF is not written.
    cutout_specs = load_cutout_specs(args)
    if cutout_specs:
        process_cutouts(B, G, R, L, cutout_specs, fits_path, wcs, args, nan_mask=nan_mask)
    else:
        # 4. Lab construction → asinh dynamic-range stretch (already applied
        #    to L and to B/G/R) → RGB → UM → write TIFF
        colorise_image(B, G, R, L, wcs, args, parser, nan_mask=nan_mask)

if __name__ == "__main__":
    main()
