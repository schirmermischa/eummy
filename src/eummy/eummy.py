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

import sys, os, glob, argparse, cv2, gc, re, tifffile
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import numexpr as ne
from importlib.metadata import version, PackageNotFoundError
from concurrent.futures import ThreadPoolExecutor


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------

# Combine default-value display with raw text formatting in --help output
class CustomHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def parse_um(values):
    """Validate and convert the --UM argument.
    Accepts either the string 'false' (disables UM) or exactly three floats
    [FWHM, strength, threshold]."""
    if len(values) == 1 and values[0].lower() == "false":
        return None
    if len(values) == 3:
        try:
            return [float(v) for v in values]
        except ValueError:
            raise argparse.ArgumentTypeError("UM values must be numeric.")
    raise argparse.ArgumentTypeError("UM must be 'false' or exactly 3 floats")


def str2bool(val):
    """Convert common truthy/falsy strings to bool for argparse type=."""
    if isinstance(val, bool):
        return val
    val = val.lower()
    if val in ("yes", "true", "t", "y", "1"):
        return True
    if val in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_arguments():
    # Read version from package metadata; fall back gracefully when running
    # the script directly without installation.
    try:
        current_version = version("eummy")
    except PackageNotFoundError:
        current_version = "dev"

    print(f"\n   eummy v{current_version} (Mischa Schirmer)\n")

    parser = argparse.ArgumentParser(
        description="Creates a colour image from Euclid MER stacks.\nRunning \"eummy \" in the directory with your images is usually sufficient.\nYou can fine-tune the result with additional command-line arguments.",
        formatter_class=CustomHelpFormatter
    )

    parser.add_argument('--version', action='version', version=f'%(prog)s {current_version}')

    # --- Input ---
    parser.add_argument("--path", default=os.getcwd(), help="Absolute or relative path to MER stacks")
    parser.add_argument("--images", nargs=4, help="Input FITS files for bands: I Y J H (in this order), if not following the MER naming convention.")

    # --- Dynamic range and tone mapping ---
    parser.add_argument("--blackwhite", nargs=2, type=float, default=[-1.3, 7000],
                        help="Min/max thresholds in linear images (J-band reference)")
    parser.add_argument("--pivot", type=float, default=0.15, help="Fraction of max value used as pivot for dynamic-range compression")
    parser.add_argument("--contrast", type=float, default=None, help="Additional contrast curve (1.0: EWS (auto), 1.6: EDS (auto), 0: off)")

    # --- Band scaling and blending ---
    parser.add_argument("--scales", nargs=4, type=float, default=[0.002336, 0.6532, 1.0000, 1.1448],
                        help="Scaling factors for bands I, Y, J, H")
    parser.add_argument("--fr", type=float, default=0.3, help="H-blending fraction for L channel")
    parser.add_argument("--fi", type=float, default=1.6, help="I-blending fraction for B channel")
    parser.add_argument("--mergeYJ", action="store_true", help="Average Y and J into green channel, uses I for blue; overrides --fi")

    # --- Colour and image quality ---
    parser.add_argument("--saturate", type=float, default=2.5, help="Colour saturation factor")
    parser.add_argument("--mask", nargs="?", default=None, const=True, type=str2bool,
                        help="Mask hot pixels; automatically applied to EWS images unless set to false")
    parser.add_argument("--useOpenCV", nargs="?", default=True, const=True, type=str2bool,
                        help="Use OpenCV for colour-space operations; high RAM usage and slower if set to False")
    parser.add_argument("--UM", nargs="*", default=["1.6", "0.75", "0.09"],
                        help="Unsharp masking: FWHM strength threshold, or 'false' to disable")

    # --- Output ---
    parser.add_argument("--output", default="TILE[id].tif", help="Output file name")
    parser.add_argument("--nthreads", type=int, default=os.cpu_count() // 2, help="Number of threads to use for parallel operations")

    # --- Cutouts ---
    # When --cutout or --cutouts is given the full mosaic TIFF is not written;
    # only the requested PNG cutouts are produced.
    cutout_help = (
        "Extract a cutout centred on a position.\n"
        "Usage:  --cutout C1 C2 W [H]\n"
        "  C1 C2  : pixel coords (e.g. 3000 4000), decimal sky coords (RA Dec in degrees),\n"
        "           or sexagesimal (e.g. 12:34:56.7 -47:30:00).\n"
        "           Auto-detected: colon → sexagesimal sky; decimal float → sky; integer → pixels.\n"
        "  W [H]  : width (and optional height) with suffix:\n"
        "           p = pixels  (e.g. 500p)\n"
        "           '  = arcmin (e.g. 5')\n"
        "           \"  = arcsec (e.g. 100\")\n"
        "           No suffix → pixels.\n"
        "  Omitting H gives a square cutout.\n"
        "  Output: 16-bit PNG named <TILE>_<RA>+<Dec>.png\n"
        "  Edge pixels outside the image are rendered black.\n"
    )
    parser.add_argument("--cutout", nargs="+", metavar="VAL",
                        help=cutout_help)
    parser.add_argument("--cutouts", nargs="+", metavar="VAL",
                        help="Text file: one cutout per line, same format as --cutout arguments;\n"
                             "  lines starting with # are ignored.  Usage: --cutouts FILE\n"
                             "FITS catalog: positions are read from the first binary table extension;\n"
                             "  RA/Dec columns are auto-detected (RIGHT_ASCENSION/DECLINATION,\n"
                             "  ALPHA_J2000/DELTA_J2000, ALPHA_ICRS/DELTA_ICRS, ALPHA/DELTA, RA/DEC).\n"
                             "  A radius must be supplied as the second argument, same suffixes as\n"
                             "  --cutout (p, ', \").  Usage: --cutouts catalog.fits 30\"\n")

    args = parser.parse_args()
    args.UM = parse_um(args.UM)
    return args, parser


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------

def extract_tileID(filename):
    """Extract the TILE<number> identifier from a MER filename and return
    it as a .tif filename, e.g. 'TILE12345678.tif'."""
    filename = os.path.basename(filename)
    match = re.search(r'(TILE\d+)\D', filename)
    if match:
        return match.group(1) + ".tif"
    else:
        return "TILE.tif"


def find_images_in_directory(path, parser):
    """Locate exactly one FITS file per band (VIS, NIR-Y, NIR-J, NIR-H) in
    path using the standard Euclid MER BGSUB-MOSAIC naming convention.
    Exits with an error if the count for any band is not exactly one."""
    vis_images   = glob.glob(os.path.join(path, "EUC_MER_BGSUB-MOSAIC-VIS*.fits"))
    nir_y_images = glob.glob(os.path.join(path, "EUC_MER_BGSUB-MOSAIC-NIR-Y*.fits"))
    nir_j_images = glob.glob(os.path.join(path, "EUC_MER_BGSUB-MOSAIC-NIR-J*.fits"))
    nir_h_images = glob.glob(os.path.join(path, "EUC_MER_BGSUB-MOSAIC-NIR-H*.fits"))

    if len(vis_images) != 1 or len(nir_y_images) != 1 or len(nir_j_images) != 1 or len(nir_h_images) != 1:
        print(f"Error: Expected exactly one image per band in {path}.\n")
        parser.print_help()
        sys.exit(1)

    tileID = extract_tileID(vis_images[0])
    return vis_images[0], nir_y_images[0], nir_j_images[0], nir_h_images[0], tileID


def extract_wcs(fits_path):
    """Read the linear WCS keywords from the primary HDU header of fits_path
    and return them as a plain dict.  The CD-matrix formalism is used
    (CD1_1 … CD2_2), which encodes pixel scale, orientation, and shear."""
    with fits.open(fits_path) as hdul:
        header = hdul[0].header
        wcs = {k: header.get(k) for k in ['EQUINOX', 'RADESYS', 'CTYPE1', 'CTYPE2',
                                           'CUNIT1', 'CUNIT2', 'CRVAL1', 'CRVAL2',
                                           'CRPIX1', 'CRPIX2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']}
    return wcs


def write_output(rgb, wcs, args):
    """Write the final uint16 RGB image as an uncompressed TIFF with the
    WCS metadata embedded, and update a 'link.tif' symlink to point to
    the new file (atomic replace to avoid a broken-link window)."""
    output_path = os.path.join(args.path, f"{args.output}")

    print(f"Writing result to {output_path}")
    tifffile.imwrite(
        output_path,
        rgb,
        metadata=wcs,
        compression=None
    )

    rgb = None
    gc.collect()

    # Atomically replace the convenience symlink
    link_name = os.path.join(args.path, "link.tif")
    tmp_link = link_name + ".tmp"
    try:
        os.symlink(args.output, tmp_link)
        os.replace(tmp_link, link_name)
    except Exception as e:
        if os.path.exists(tmp_link):
            os.unlink(tmp_link)
        raise


# ---------------------------------------------------------------------------
# Image processing pipeline
#
# Call order: rescale_and_blend → asinh_scale_and_normalise →
#             contrast_adjustment → colorise_L → write_output
# ---------------------------------------------------------------------------

def repair_bad_pixels(B, G, R, L, args):
    """Replace missing and anomalous pixel values before colour composition.

    Four repair passes are applied in sequence:

    1. Bad NISP pixels (zero in any NIR band): replaced with the VIS
       luminance value L so the pixel contributes greyscale rather than a
       colour artefact.
    2. Bad VIS pixels (L == 0 while all NIR bands are valid): replaced with
       the mean of the three NIR channels.
    3. Hot pixels (automatic, EWS only): pixels that are both absolutely
       bright and anomalously bright relative to their neighbours across
       channels are replaced with the local cross-channel average.  The VIS
       threshold is intentionally high to avoid clipping compact star cores.
    4. Any pixel still zero in any channel after the above is set to a large
       value (1e5) so that it renders as white rather than black after the
       arcsinh stretch.
    """
    print("Repairing bad pixels")

    mask = np.empty(L.shape, dtype=bool)

    # 1. Bad NISP: any NIR band zero while VIS is valid → use VIS luminance
    ne.evaluate("((B==0) | (G==0) | (R==0)) & (L!=0)", out=mask)
    ne.evaluate("where(mask, L, B)", out=B)
    ne.evaluate("where(mask, L, G)", out=G)
    ne.evaluate("where(mask, L, R)", out=R)

    # 2. Bad VIS: L zero while all NIR bands valid → use NIR mean
    avg_nir = "(B + G + R) / 3.0"
    ne.evaluate(f"where((B!=0) & (G!=0) & (R!=0) & (L==0), {avg_nir}, L)", out=L)

    # 3. Hot pixel masking — applied automatically to large (EWS) images or
    #    when explicitly requested.  Pixels are flagged as hot if they exceed
    #    an absolute threshold AND are more than thresh2× brighter than the
    #    cross-channel average; replaced by that average.
    height, width = L.shape
    if (height > 15000 and width > 15000 and args.mask is not False) or args.mask is True:

        # thresh4 is high for VIS because the compact PSF produces genuinely
        # bright single-pixel star cores that should not be masked.
        thresh1, thresh2, thresh3, thresh4 = 5, 5, 3, 20

        ne.evaluate("(B > th1) & (B > th2 * (G+R+L)/3)",
                    local_dict={'B': B, 'G': G, 'R': R, 'L': L, 'th1': thresh1, 'th2': thresh2}, out=mask)
        ne.evaluate("where(mask, (G+R+L)/3, B)", out=B)

        ne.evaluate("(G > th1) & (G > th2 * (B+R+L)/3)",
                    local_dict={'B': B, 'G': G, 'R': R, 'L': L, 'th1': thresh1, 'th2': thresh2}, out=mask)
        ne.evaluate("where(mask, (B+R+L)/3, G)", out=G)

        ne.evaluate("(R > th1) & (R > th2 * (B+G+L)/3)",
                    local_dict={'B': B, 'G': G, 'R': R, 'L': L, 'th1': thresh1, 'th2': thresh2}, out=mask)
        ne.evaluate("where(mask, (B+G+L)/3, R)", out=R)

        ne.evaluate("(L > th3) & (L > th4 * (B+G+R)/3)",
                    local_dict={'B': B, 'G': G, 'R': R, 'L': L, 'th3': thresh3, 'th4': thresh4}, out=mask)
        ne.evaluate("where(mask, (B+G+R)/3, L)", out=L)

    # 4. Any remaining zero in any channel → set to white (saturated)
    ne.evaluate("(B==0) | (G==0) | (R==0) | (L==0)", out=mask)
    B[mask] = 1e5
    G[mask] = 1e5
    R[mask] = 1e5


def rescale_and_blend(args, parser):
    """Load the four Euclid MER band images and blend them into the B, G, R,
    and L channels used by the downstream colour pipeline.

    Band-to-channel mapping
    -----------------------
    B (blue)      : weighted blend of Y and I  (controlled by --fi)
    G (green)     : J-band, or average of Y+J with --mergeYJ
    R (red)       : H-band
    L (luminance) : I-band with a small adaptive H contribution (--fr),
                    softened by an exponential weight that prevents bright
                    H-band sources from overwhelming faint VIS structure.

    The four FITS files are read in parallel to overlap I/O latency.
    Per-band flux scaling (--scales) is applied in-place via numexpr before
    any blending.  Bad pixels are repaired before blending so that artefacts
    are not spread into adjacent channels.

    Returns (B, G, R, L, wcs_dict, fits_path_of_i_band).
    """
    if not os.path.isdir(args.path):
        print(f"Error: Directory '{args.path}' does not exist.")
        sys.exit(1)

    if args.images:
        i_band, y_band, j_band, h_band = args.images
        tileID = extract_tileID(i_band)
    else:
        i_band, y_band, j_band, h_band, tileID = find_images_in_directory(args.path, parser)

    # Resolve final output filename before any processing
    if args.output == "TILE[id].tif":
        args.output = tileID

    si, sy, sj, sh = args.scales

    print("Processing FITS images")

    def load_and_prep(file_path):
        # astropy.io.fits preserves FITS row order (row 0 = bottom of sky).
        # np.asanyarray returns a view when the data is already float32.
        with fits.open(file_path) as hdul:
            data = hdul[0].data
            return np.asanyarray(data, dtype=np.float32)

    # Read all four bands in parallel to overlap disk latency
    files = [f if os.path.isabs(f) else os.path.join(args.path, f) for f in [y_band, j_band, h_band, i_band]]
    with ThreadPoolExecutor(max_workers=4) as executor:
        y_data, j_data, h_data, i_data = list(executor.map(load_and_prep, files))

    # Extract and store WCS from the VIS (I) band header for use in output
    fits_i_path = i_band if os.path.isabs(i_band) else os.path.join(args.path, i_band)
    wcs = extract_wcs(fits_i_path)

    # Apply per-band flux scaling in-place (skip bands where scale == 1.0)
    for data, scale, name in [(y_data, sy, "Y"), (j_data, sj, "J"),
                               (h_data, sh, "H"), (i_data, si, "VIS")]:
        if scale != 1.0:
            ne.evaluate("data / scale", local_dict={'data': data, 'scale': scale}, out=data)

    repair_bad_pixels(y_data, j_data, h_data, i_data, args)

    # B: Y + fi * I, normalised — blends NIR-Y with VIS for a blue-ish channel
    fi = args.fi
    if args.mergeYJ:
        B = i_data
    else:
        B = ne.evaluate("(y_data + i_data * fi) / (1.0 + fi)",
                        local_dict={'y_data': y_data, 'i_data': i_data, 'fi': fi})

    # G: J-band alone, or average of Y+J with --mergeYJ
    if args.mergeYJ:
        G = ne.evaluate("(y_data + j_data) * 0.5", local_dict={'y_data': y_data, 'j_data': j_data})
    else:
        G = j_data

    # L: I-band with an adaptive H-band contribution.  The exponential weight
    #    exp(-0.2*|I|) tapers the H contribution in bright regions, preserving
    #    fine VIS structure in galaxy cores and stellar halos.
    fr = args.fr
    if fr > 0:
        L = ne.evaluate("(i_data + fr * exp(-0.2*abs(i_data)) * h_data) / (1.0 + fr * exp(-0.2*abs(i_data)))",
                        local_dict={'i_data': i_data, 'h_data': h_data, 'fr': fr})
    else:
        L = i_data

    # R: H-band directly
    R = h_data

    # Release temporaries no longer aliased by B, G, R, L
    y_data = None
    if fr > 0: i_data = None
    gc.collect()

    return B, G, R, L, wcs, fits_i_path


def asinh_scale_and_normalise(B, G, R, L, args):
    """Compress dynamic range with an arcsinh stretch and normalise to [0, 1].

    The arcsinh function suppresses bright outliers while preserving detail
    in faint regions.  The pivot parameter controls the linear-to-nonlinear
    transition point.  Black and white levels are set by args.blackwhite and
    applied as a linear rescaling after the stretch.  All four channels are
    processed in-place in a single numexpr pass to minimise memory traffic.
    """
    print(f"Dynamic-range compression (pivot {args.pivot}) and normalisation [{args.blackwhite[0]}, {args.blackwhite[1]}]")
    p = args.pivot
    minval, maxval = args.blackwhite
    black = np.arcsinh(p * minval)
    white = np.arcsinh(p * maxval)
    scale = 1.0 / (white - black)

    for ch in [B, G, R, L]:
        ne.evaluate("(arcsinh(p * ch) - black) * scale",
                    local_dict={'ch': ch, 'p': p, 'black': black, 'scale': scale},
                    out=ch)


def contrast_adjustment(L, args):
    """Apply a mild S-curve contrast boost to the luminance channel L.

    The curve is a 3rd-order polynomial  y = 0.5707x³ - 1.8298x² + 2.2592x
    that lifts mid-tones while leaving the black and white points anchored.
    The strength is controlled by args.contrast, which scales the deviation
    from the identity line.  EDS images default to 1.6; EWS to 1.0.
    The operation is done in-place via numexpr.
    """
    if args.contrast == 0:
        return

    height, width = L.shape
    # Auto-select strength based on image size (EDS = 10200×10200, EWS = larger)
    if args.contrast is None:
        if height == 10200 and width == 10200:
            args.contrast = 1.6
            print(f"Enhancing contrast by {args.contrast} (default for EDS)")
        else:
            args.contrast = 1.0
            print(f"Enhancing contrast by {args.contrast} (default for EWS)")
    else:
        print(f"Enhancing contrast by {args.contrast}")

    c = args.contrast
    ne.evaluate(
        "c * (0.5707 * L**3 - 1.8298 * L**2 + 2.2592 * L - L) + L",
        local_dict={'L': L, 'c': c}, out=L)


# ---------------------------------------------------------------------------
# Colour-space pipeline: RGB → CIELab → (luminance swap + saturation) → RGB
#
# Both implementations perform the same logical operation:
#   1. Convert the blended B/G/R channels to CIELab.
#   2. Replace the Lab L* channel with the separately computed luminance L,
#      which carries finer VIS detail than the colour channels alone.
#   3. Scale the a* and b* chrominance channels by args.saturate.
#   4. Convert back to RGB.
#
# The OpenCV path is ~5× faster because cv2.cvtColor uses a look-up table
# for the gamma power law, among others; the manual path is kept for 
# reference and as a low-dependency fallback.
# ---------------------------------------------------------------------------

def rgb_lab_rgb_OpenCV(rgb, L, args):
    """Luminance swap and saturation boost via OpenCV's CIELab conversion."""
    print("Color-space operations")

    # RGB → Lab (OpenCV uses float32 input in range [0, 1])
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    rgb = None; gc.collect()

    # Scale a* and b* chrominance channels in-place, clamped to valid Lab range
    s = args.saturate
    for i in [1, 2]:
        ch_view = lab[:, :, i]
        ne.evaluate("where(ch*s > 127, 127, where(ch*s < -128, -128, ch*s))",
                    local_dict={'ch': ch_view, 's': s}, out=ch_view)

    # Replace the L* channel (range 0–100) with the pre-computed luminance
    ne.evaluate("L * 100.0", local_dict={'L': L}, out=lab[:, :, 0])
    L = None; gc.collect()

    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    lab = None
    gc.collect()

    return rgb


def rgb_lab_rgb_manual(rgb, L, args):
    """Luminance swap and saturation boost via a fully manual CIELab pipeline.
    Slower than the OpenCV version (~5×) but has no OpenCV dependency."""
    print("Color-space operations")

    # --- RGB → Lab ---

    # Step 1: Linearise sRGB (undo gamma) before converting to XYZ
    ne.evaluate("where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)",
                local_dict={'rgb': rgb}, out=rgb)

    # Views into the three colour planes (no copy)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    # Step 2: Linear RGB → XYZ (D65, sRGB primaries)
    X = ne.evaluate("0.4124564*r + 0.3575761*g + 0.1804375*b")
    Y = ne.evaluate("0.2126729*r + 0.7151522*g + 0.0721750*b")
    Z = ne.evaluate("0.0193339*r + 0.1191920*g + 0.9503041*b")
    rgb = None
    gc.collect()

    # Step 3: Normalise by D65 white point
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    ne.evaluate("X / Xn", out=X)
    ne.evaluate("Y / Yn", out=Y)
    ne.evaluate("Z / Zn", out=Z)

    # Step 4: Apply the CIE f-function (cube root with linear extension below threshold)
    eps   = 0.008856    # (6/29)³
    kappa = 7.7870      # 1/3 · (29/6)²
    ne.evaluate("where(X > eps, X ** (1.0/3.0), kappa * X + 0.137931)",
                local_dict={'X': X, 'eps': eps, 'kappa': kappa}, out=X)
    ne.evaluate("where(Y > eps, Y ** (1.0/3.0), kappa * Y + 0.137931)",
                local_dict={'Y': Y, 'eps': eps, 'kappa': kappa}, out=Y)
    ne.evaluate("where(Z > eps, Z ** (1.0/3.0), kappa * Z + 0.137931)",
                local_dict={'Z': Z, 'eps': eps, 'kappa': kappa}, out=Z)

    # Step 5: Compute a* and b* chrominance, apply saturation, clamp to valid range
    s = args.saturate
    a_sat = ne.evaluate("500.0 * (X - Y) * s", local_dict={'X': X, 'Y': Y, 's': s})
    b_sat = ne.evaluate("200.0 * (Y - Z) * s", local_dict={'Y': Y, 'Z': Z, 's': s})
    ne.evaluate("where(a_sat >  127,  127, where(a_sat < -128, -128, a_sat))", local_dict={'a_sat': a_sat}, out=a_sat)
    ne.evaluate("where(b_sat >  127,  127, where(b_sat < -128, -128, b_sat))", local_dict={'b_sat': b_sat}, out=b_sat)

    # Use the passed-in luminance L instead of the computed L* channel
    light = ne.evaluate("L * 100.0", local_dict={'L': L})

    # --- Lab → RGB ---

    # Step 6: Lab → XYZ (inverse f-function; note eps is 6/29, not (6/29)³)
    eps = 0.206896   # 6/29
    ne.evaluate("(light + 16.0) / 116.0", local_dict={'light': light}, out=Y)
    ne.evaluate("Y + a_sat / 500.0", local_dict={'Y': Y, 'a_sat': a_sat}, out=X)
    ne.evaluate("Y - b_sat / 200.0", local_dict={'Y': Y, 'b_sat': b_sat}, out=Z)

    ne.evaluate("where(Y > eps, Y**3, (Y - 0.137931) / kappa)",
                local_dict={'Y': Y, 'eps': eps, 'kappa': kappa}, out=Y)
    ne.evaluate("where(X > eps, X**3, (X - 0.137931) / kappa)",
                local_dict={'X': X, 'eps': eps, 'kappa': kappa}, out=X)
    ne.evaluate("where(Z > eps, Z**3, (Z - 0.137931) / kappa)",
                local_dict={'Z': Z, 'eps': eps, 'kappa': kappa}, out=Z)

    # Re-apply D65 white point
    ne.evaluate("X * Xn", out=X)
    ne.evaluate("Y * Yn", out=Y)
    ne.evaluate("Z * Zn", out=Z)

    # Step 7: XYZ → linear RGB (inverse sRGB matrix)
    r = ne.evaluate("3.2404542*X - 1.5371385*Y - 0.4985314*Z")
    g = ne.evaluate("-0.9692660*X + 1.8760108*Y + 0.0415560*Z")
    b = ne.evaluate("0.0556434*X - 0.2040259*Y + 1.0572252*Z")
    X = Y = Z = a_sat = b_sat = None
    gc.collect()

    # Step 8: Apply sRGB gamma and clamp to [0, 1]
    ne.evaluate("where(r > 0.0031308, 1.055 * r**(1.0/2.4) - 0.055, 12.92 * r)", out=r)
    ne.evaluate("where(g > 0.0031308, 1.055 * g**(1.0/2.4) - 0.055, 12.92 * g)", out=g)
    ne.evaluate("where(b > 0.0031308, 1.055 * b**(1.0/2.4) - 0.055, 12.92 * b)", out=b)

    ne.evaluate("where(r > 1, 1, where(r < 0, 0, r))", out=r)
    ne.evaluate("where(g > 1, 1, where(g < 0, 0, g))", out=g)
    ne.evaluate("where(b > 1, 1, where(b < 0, 0, b))", out=b)
    rgb = np.stack([r, g, b], axis=-1)
    r = g = b = None

    return rgb


def unsharp_mask(image, radius=1.6, strength=0.75, threshold=0.09):
    """Sharpen image by subtracting a Gaussian-blurred version of itself.

    Only pixels where the local contrast (|image - blurred|) exceeds
    threshold are sharpened, which avoids amplifying noise in flat regions.
    The result is clipped to [0, 1] in-place.  All operations are fused into
    single numexpr passes to avoid intermediate allocations.
    """
    print(f"Unsharp masking with {radius, strength, threshold}")
    ksize = max(3, int(2*round(radius*2.5)+1))
    blurred = cv2.GaussianBlur(image, (ksize, ksize), radius)

    ne.evaluate("where(abs(image - blurred) >= threshold, image + strength * (image - blurred), image)",
                local_dict={'image': image, 'blurred': blurred, 'threshold': threshold, 'strength': strength},
                out=image)

    ne.evaluate("where(image > 1, 1, where(image < 0, 0, image))", out=image)


def colorise_L(B, G, R, L, wcs, args, parser):
    """Combine the four float32 channels into a colour image and write the TIFF.

    The B, G, R channels carry chrominance information; L carries luminance.
    They are merged in CIELab space: chrominance comes from B/G/R, while the
    Lab L* channel is replaced by the separately blended luminance L.  This
    preserves fine VIS detail in the brightness while the NIR bands provide
    natural colour.

    Peak RAM here is approximately:
      ~5.6 GB for the four input channels (19200×19200 float32)
      + ~3.7 GB for the interleaved rgb array
      + temporary XYZ arrays in the manual Lab path (~11 GB peak total)
    The input channels are freed as they are consumed into rgb.
    """
    # Pack the three colour channels into a contiguous H×W×3 float32 array.
    # Channels are freed immediately after assignment to minimise peak RAM.
    rgb = np.empty((R.shape[0], R.shape[1], 3), dtype=np.float32)
    rgb[:, :, 0] = R;  R = None
    rgb[:, :, 1] = G;  G = None
    rgb[:, :, 2] = B;  B = None
    gc.collect()

    if args.useOpenCV:
        rgb = rgb_lab_rgb_OpenCV(rgb, L, args)
    else:
        rgb = rgb_lab_rgb_manual(rgb, L, args)
    L = None
    gc.collect()

    if args.UM is not None:
        fwhm, strength, threshold = args.UM
        unsharp_mask(rgb, fwhm, strength, threshold)

    # Scale float32 [0,1] → uint16 [0,65535] and flip vertically so that
    # north is up in the output (FITS row 0 is at the bottom; the flip puts
    # it at the end of the file, which is the top in image-viewer convention).
    rgb_out = np.empty(rgb.shape, dtype=np.uint16)
    np.multiply(rgb[::-1], 65535, out=rgb_out, casting='unsafe')
    rgb = None
    gc.collect()

    write_output(rgb_out, wcs, args)
    rgb_out = None
    gc.collect()


# ---------------------------------------------------------------------------
# Cutout support
# ---------------------------------------------------------------------------

def _parse_dim(token):
    """Parse a dimension token into (value, unit).
    Accepted suffixes:  p or none → pixels,  ' → arcmin,  " → arcsec."""
    token = token.strip()
    if token.endswith("p"):
        return float(token[:-1]), 'p'
    elif token.endswith("'"):
        return float(token[:-1]), 'arcmin'
    elif token.endswith('"'):
        return float(token[:-1]), 'arcsec'
    else:
        return float(token), 'p'     # bare number → pixels


def _coord_is_sky(s):
    """Return True if s looks like a sky coordinate.
    Rules:
      - contains ':'              → sexagesimal (always sky)
      - contains '.' or 'e'/'E'  → decimal float → sky (RA/Dec)
      - bare integer              → pixel coordinate
      - non-numeric               → assume sky, let SkyCoord report the error
    """
    if ':' in s:
        return True
    if '.' in s or 'e' in s.lower():
        return True
    try:
        int(s)
        return False    # bare integer → pixel
    except ValueError:
        return True     # non-numeric → try as sky


def _parse_cutout_tokens(tokens):
    """Parse 3 or 4 tokens [C1, C2, W] or [C1, C2, W, H] into a dict:
      { 'c1': str, 'c2': str, 'w_val': float, 'w_unit': str,
        'h_val': float, 'h_unit': str, 'is_sky': bool }
    Raises ValueError with a descriptive message on bad input."""
    if len(tokens) < 3 or len(tokens) > 4:
        raise ValueError(f"Expected 3 or 4 values (C1 C2 W [H]), got {len(tokens)}: {tokens}")

    c1, c2 = tokens[0], tokens[1]
    w_val, w_unit = _parse_dim(tokens[2])
    if len(tokens) == 4:
        h_val, h_unit = _parse_dim(tokens[3])
    else:
        h_val, h_unit = w_val, w_unit

    is_sky = _coord_is_sky(c1) or _coord_is_sky(c2)

    return dict(c1=c1, c2=c2,
                w_val=w_val, w_unit=w_unit,
                h_val=h_val, h_unit=h_unit,
                is_sky=is_sky)


def _dim_to_pixels(val, unit, fits_path):
    """Convert a dimension (val, unit) to integer pixels.
    For angular units the pixel scale is derived from the WCS CD matrix;
    fits_path is only opened when needed."""
    if unit == 'p':
        return max(1, int(round(val)))

    # Need pixel scale in arcsec/pixel — derive from WCS
    with fits.open(fits_path) as hdul:
        w = WCS(hdul[0].header)
    ps = w.proj_plane_pixel_scales()   # returns Quantity array
    pix_scale_arcsec = (ps[0].to(u.arcsec).value + ps[1].to(u.arcsec).value) / 2

    arcsec = val * 60.0 if unit == 'arcmin' else val
    return max(1, int(round(arcsec / pix_scale_arcsec)))


def _sky_to_pixel(c1_str, c2_str, fits_path):
    """Convert sky coordinates (sexagesimal or decimal degrees) to
    0-based pixel (col, row) using the FITS WCS.  Returns (col, row)."""
    with fits.open(fits_path) as hdul:
        w = WCS(hdul[0].header)

    # SkyCoord handles both sexagesimal and decimal transparently
    if ':' in c1_str or ':' in c2_str:
        coord = SkyCoord(c1_str, c2_str, unit=(u.hourangle, u.deg))
    else:
        coord = SkyCoord(float(c1_str), float(c2_str), unit=u.deg)

    # world_to_pixel returns (x, y) = (col, row), 0-based
    col, row = w.world_to_pixel(coord)
    return float(col), float(row)


def _pixel_to_sky(col, row, fits_path):
    """Convert 0-based pixel (col, row) to (RA, Dec) in decimal degrees."""
    with fits.open(fits_path) as hdul:
        w = WCS(hdul[0].header)
    coord = w.pixel_to_world(col, row)
    return float(coord.ra.deg), float(coord.dec.deg)


def _cutout_filename(base_tif, ra_deg, dec_deg):
    """Build the PNG filename from the TIFF base name and sky position.
    e.g. TILE12345678_100.345655-47.567834.png
    Precision: 6 decimal degrees ≈ 0.036 arcsec, well below the 0.1" requirement."""
    stem = os.path.splitext(os.path.basename(base_tif))[0]
    sign = '+' if dec_deg >= 0 else ''
    return f"{stem}_{ra_deg:.6f}{sign}{dec_deg:.6f}.png"


def _crop_channel(ch, col_c, row_c, w_px, h_px):
    """Crop a 2-D float32 channel centred on (col_c, row_c).

    col_c, row_c are 0-based FITS pixel coordinates.  astropy.io.fits loads
    data with row 0 at array index 0 (FITS convention, origin at lower-left),
    so FITS row N is directly at ch[N] — no axis flip needed here.

    Returns (crop, dy0, dx0) where dy0/dx0 are the offsets into the
    black-padded canvas so the caller can place the crop correctly when the
    window is partially outside the image.
    """
    img_h, img_w = ch.shape

    half_w = w_px / 2.0
    half_h = h_px / 2.0

    x0 = int(np.floor(col_c - half_w))
    x1 = x0 + w_px
    y0 = int(np.floor(row_c - half_h))
    y1 = y0 + h_px

    # Clamp to image bounds
    x0c = max(x0, 0);  x1c = min(x1, img_w)
    y0c = max(y0, 0);  y1c = min(y1, img_h)

    if x0c >= x1c or y0c >= y1c:
        return None, 0, 0   # fully outside

    crop = ch[y0c:y1c, x0c:x1c]   # view, no copy
    dx0  = x0c - x0
    dy0  = y0c - y0
    return crop, dy0, dx0


def _render_cutout_crop(B, G, R, L, col_c, row_c, w_px, h_px, args):
    """Extract and fully render a single cutout from the four float32 channels.

    The colour pipeline (Lab round-trip, optional UM) is applied only to the
    small crop, keeping peak memory proportional to the cutout size rather
    than the full image.

    Returns a (h_px, w_px, 3) uint16 array in BGR order ready for
    cv2.imwrite, or None if the centre is entirely outside the image.
    """
    # Crop each channel; placement offsets are identical across channels
    # since all bands share the same pixel grid after blending.
    B_crop, dy0, dx0 = _crop_channel(B, col_c, row_c, w_px, h_px)
    if B_crop is None:
        return None

    G_crop, _, _ = _crop_channel(G, col_c, row_c, w_px, h_px)
    R_crop, _, _ = _crop_channel(R, col_c, row_c, w_px, h_px)
    L_crop, _, _ = _crop_channel(L, col_c, row_c, w_px, h_px)

    ch, cw = B_crop.shape

    def _pad(crop):
        # If the crop fills the full canvas no padding is needed; copy to
        # ensure the Lab functions can write in-place without aliasing the
        # original channel arrays.
        if crop.shape == (h_px, w_px):
            return crop.copy()
        canvas = np.zeros((h_px, w_px), dtype=np.float32)
        canvas[dy0:dy0+ch, dx0:dx0+cw] = crop
        return canvas

    Bc = _pad(B_crop)
    Gc = _pad(G_crop)
    Rc = _pad(R_crop)
    Lc = _pad(L_crop)

    # Stack into H×W×3 float32 and run the colour pipeline on the small crop
    rgb = np.empty((h_px, w_px, 3), dtype=np.float32)
    rgb[:, :, 0] = Rc
    rgb[:, :, 1] = Gc
    rgb[:, :, 2] = Bc
    Bc = Gc = Rc = None

    if args.useOpenCV:
        rgb = rgb_lab_rgb_OpenCV(rgb, Lc, args)
    else:
        rgb = rgb_lab_rgb_manual(rgb, Lc, args)
    Lc = None

    if args.UM is not None:
        fwhm, strength, threshold = args.UM
        unsharp_mask(rgb, fwhm, strength, threshold)

    # Scale to uint16. The crop is in FITS row order (row 0 = bottom), so flip
    # vertically to produce a PNG consistent with the full TIFF output / FITS file
    out = np.empty((h_px, w_px, 3), dtype=np.uint16)
    np.multiply(rgb[::-1], 65535, out=out, casting='unsafe')
    rgb = None

    # cv2.imwrite expects BGR
    return out[:, :, ::-1]


def _load_fits_catalog(cat_path, radius_token):
    """Read RA/Dec positions from a FITS binary table and return a list of
    token lists suitable for process_cutouts().

    Column name priority (tried in order):
      RIGHT_ASCENSION / DECLINATION
      ALPHA_J2000     / DELTA_J2000
      ALPHA_ICRS      / DELTA_ICRS
      ALPHA           / DELTA
      RA              / DEC

    All coordinates are returned as decimal degree strings (sky coords).
    Exits with an error if no recognised column pair is found.
    """
    RA_NAMES  = ['RIGHT_ASCENSION', 'ALPHA_J2000', 'ALPHA_ICRS', 'ALPHA', 'RA']
    DEC_NAMES = ['DECLINATION',     'DELTA_J2000', 'DELTA_ICRS', 'DELTA', 'DEC']

    with fits.open(cat_path) as hdul:
        # Find the first binary table extension
        table = None
        for hdu in hdul[1:]:
            if hasattr(hdu, 'columns'):
                table = hdu
                break
        if table is None:
            print(f"ERROR: no binary table extension found in '{cat_path}'")
            sys.exit(1)

        col_names = [c.upper() for c in table.columns.names]

        ra_col = dec_col = None
        for ra_try, dec_try in zip(RA_NAMES, DEC_NAMES):
            if ra_try in col_names and dec_try in col_names:
                # Use original-case names for data access
                orig = table.columns.names
                ra_col  = orig[col_names.index(ra_try)]
                dec_col = orig[col_names.index(dec_try)]
                break

        if ra_col is None:
            tried = ', '.join(f'{r}/{d}' for r, d in zip(RA_NAMES, DEC_NAMES))
            print(f"ERROR: no recognised RA/Dec columns in '{cat_path}'.\n"
                  f"  Tried (case-insensitive): {tried}\n"
                  f"  Available columns: {', '.join(table.columns.names)}")
            sys.exit(1)

        print(f"  Using catalog columns: {ra_col}, {dec_col} ({len(table.data)} rows)")
        ra_vals  = table.data[ra_col].astype(float)
        dec_vals = table.data[dec_col].astype(float)

    # Build token lists identical to what the text-file path produces
    return [[f"{ra:.8f}", f"{dec:.8f}", radius_token]
            for ra, dec in zip(ra_vals, dec_vals)]


def load_cutout_specs(args):
    """Return a list of token lists for process_cutouts(), sourced from
    exactly one of --cutout or --cutouts (they are mutually exclusive).

    --cutout  C1 C2 W [H]   -- single cutout on the command line
    --cutouts FILE           -- plain text file, one cutout per line
    --cutouts FILE RADIUS    -- FITS binary table catalog with a uniform radius

    The file type for --cutouts is determined by attempting fits.open();
    if that succeeds it is treated as a FITS catalog regardless of extension.
    """
    if args.cutout and args.cutouts:
        print("ERROR: --cutout and --cutouts are mutually exclusive; use one or the other.")
        sys.exit(1)

    if args.cutout:
        return [args.cutout]

    if args.cutouts:
        cutouts_file   = args.cutouts[0]
        cutouts_radius = args.cutouts[1] if len(args.cutouts) > 1 else None

        if len(args.cutouts) > 2:
            print(f"ERROR: --cutouts takes 1 or 2 arguments (FILE [RADIUS]), got {len(args.cutouts)}")
            sys.exit(1)

        # Sniff: try opening as FITS; fall back to plain text
        is_fits = False
        try:
            with fits.open(cutouts_file):
                is_fits = True
        except Exception:
            pass

        if is_fits:
            if cutouts_radius is None:
                print("ERROR: --cutouts with a FITS catalog requires a radius as the second argument\n"
                      "  e.g.  --cutouts catalog.fits 30\"")
                sys.exit(1)
            return _load_fits_catalog(cutouts_file, cutouts_radius)
        else:
            if cutouts_radius is not None:
                print(f"WARNING: --cutouts radius argument '{cutouts_radius}' is ignored for text files")
            specs = []
            try:
                with open(cutouts_file) as fh:
                    for lineno, line in enumerate(fh, 1):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        tokens = line.split()
                        if len(tokens) < 3 or len(tokens) > 4:
                            print(f"  WARNING: {cutouts_file}:{lineno}: expected 3-4 tokens, "
                                  f"got {len(tokens)} — skipping")
                            continue
                        specs.append(tokens)
            except OSError as e:
                print(f"ERROR: cannot open cutouts file '{cutouts_file}': {e}")
                sys.exit(1)
            return specs

    return []


def process_cutouts(B, G, R, L, cutout_specs, fits_path, args):
    """Process all cutout specifications directly from the four float32 channel
    arrays produced after asinh scaling and contrast adjustment — before the
    memory-intensive full-image Lab conversion in colorise_L.

    Each cutout is rendered independently through the colour pipeline on its
    small crop, so peak RAM is ~5.6 GB (the four channels) plus one tiny crop,
    rather than the ~21 GB peak of the full pipeline, for a EWS tile.

    cutout_specs: list of token lists, e.g.
        [['3000', '4000', '500p'], ['12:34:56', '-47:30:00', "5'"]]
    fits_path: path to the I-band FITS file used for WCS.
    """
    print(f"\nExtracting {len(cutout_specs)} cutout(s)...")
    img_h, img_w = L.shape

    for tokens in cutout_specs:
        desc = ' '.join(tokens)
        try:
            spec = _parse_cutout_tokens(tokens)
        except ValueError as e:
            print(f"  WARNING: skipping '{desc}': {e}")
            continue

        # Resolve pixel centre (0-based FITS coords)
        try:
            if spec['is_sky']:
                # astropy world_to_pixel() returns 0-based coords natively
                col_c, row_c = _sky_to_pixel(spec['c1'], spec['c2'], fits_path)
            else:
                # User supplies 1-based FITS pixel coords; convert to 0-based.
                col_c = float(spec['c1']) - 1.0
                row_c = float(spec['c2']) - 1.0
        except Exception as e:
            print(f"  WARNING: skipping '{desc}': coordinate conversion failed: {e}")
            continue

        # Convert dimensions to pixels
        try:
            w_px = _dim_to_pixels(spec['w_val'], spec['w_unit'], fits_path)
            h_px = _dim_to_pixels(spec['h_val'], spec['h_unit'], fits_path)
        except Exception as e:
            print(f"  WARNING: skipping '{desc}': dimension conversion failed: {e}")
            continue

        # Always derive RA/Dec for the filename (even if input was in pixels)
        try:
            ra_deg, dec_deg = _pixel_to_sky(col_c, row_c, fits_path)
        except Exception as e:
            print(f"  WARNING: skipping '{desc}': sky coordinate lookup failed: {e}")
            continue

        # Sanity-check centre is inside the image
        if col_c < 0 or col_c >= img_w or row_c < 0 or row_c >= img_h:
            print(f"  WARNING: skipping '{desc}': centre ({col_c:.1f}, {row_c:.1f}) "
                  f"is outside the image ({img_w}×{img_h})")
            continue

        bgr_out = _render_cutout_crop(B, G, R, L, col_c, row_c, w_px, h_px, args)
        if bgr_out is None:
            print(f"  WARNING: skipping '{desc}': cutout window entirely outside image")
            continue

        fname    = _cutout_filename(args.output, ra_deg, dec_deg)
        out_path = os.path.join(args.path, fname)
        cv2.imwrite(out_path, bgr_out)
        print(f"  Cutout saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args, parser = parse_arguments()
    ne.set_num_threads(args.nthreads)
    cv2.setNumThreads(args.nthreads)

    B, G, R, L, wcs, fits_path = rescale_and_blend(args, parser)
    asinh_scale_and_normalise(B, G, R, L, args)
    contrast_adjustment(L, args)

    # Cutouts are extracted here, before colorise_L, while only the four
    # float32 channel arrays are live (~5.6 GB for a 19200×19200 EWS image).
    # Each cutout runs the full colour pipeline on its small crop only, so
    # peak RAM during cutout extraction is proportional to the cutout size.
    # When cutouts are requested the full mosaic TIFF is not written.
    cutout_specs = load_cutout_specs(args)
    if cutout_specs:
        process_cutouts(B, G, R, L, cutout_specs, fits_path, args)
    else:
        colorise_L(B, G, R, L, wcs, args, parser)

if __name__ == "__main__":
    main()
