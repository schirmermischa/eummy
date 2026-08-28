#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eummy_io.py -- File and header I/O for eummy: argument parsing, FITS
file discovery, header/WCS reading, cutout-spec parsing, and TIFF output.

No pixel-value math lives here -- this module only reads/writes files and
parses configuration. Depended on by eummy_fluxspace.py and
eummy_colorspace.py; has no dependency on either.
"""

import sys, os, glob, argparse, re, gc, shlex, warnings
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import tifffile
from importlib.metadata import version, PackageNotFoundError

# astropy.wcs emits FITSFixedWarning whenever it silently normalises a
# non-standard but unambiguous header value (e.g. deriving DATE-OBS from
# MJD-OBS). This is informational, not actionable, and fires on essentially
# every WCS() call in this module -- suppressed globally rather than at each
# of the several call sites.
warnings.filterwarnings('ignore', category=FITSFixedWarning)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

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

def parse_arguments():
    try:
        current_version = version("eummy")
    except PackageNotFoundError:
        current_version = "dev"

    print(f"\n   eummy v{current_version} (Mischa Schirmer)\n")

    parser = argparse.ArgumentParser(
        description="Creates a colour image from Euclid MER stacks.\nRunning \"eummy\" in the directory with your images is usually sufficient.",
        formatter_class=CustomHelpFormatter
    )

    parser.add_argument('--version', action='version', version=f'%(prog)s {current_version}')

    # --- Input ---
    parser.add_argument("--path", default=os.getcwd(),
                        help="Absolute or relative path to MER stacks")
    parser.add_argument("--images", nargs=4,
                        help="Input FITS files for bands: I Y J H (in this order),\n"
                             "if not following the MER naming convention.")

    # --- Black/white points ---
    parser.add_argument("--blackwhite", nargs=2, type=float, default=[-1.3, 7000],
                        help="Min/max flux thresholds for normalisation to [0,1].\n"
                             "Applied AFTER photometric zero-point correction and flux scaling\n"
                             "(--scales) -- i.e. in the common, zero-point-independent flux scale\n"
                             "that check_zeropoints normalises every band onto (J-band reference,\n"
                             "zp=30.1), not raw pixel counts. As long as zero-point correction is\n"
                             "active (the default; not --no-zp-check), these values do NOT need to\n"
                             "be recalibrated per image regardless of that image's actual zero-\n"
                             "point -- that consistency is the whole point of the correction.\n"
                             "With --no-zp-check, --scales is used exactly as given (uncorrected),\n"
                             "so these values then act directly on scale-divided raw flux and may\n"
                             "need per-image adjustment.")

    # --- Band scaling and blending ---
    # DR1 values
    parser.add_argument("--scales", nargs=4, type=float, default=[0.002078, 0.5890, 1.0000, 1.1061],
                        help="Photometric scaling factors for bands I Y J H")
    parser.add_argument("--no-zp-check", action="store_true",
                        help="Use --scales exactly as given, without adjusting for any\n"
                             "MAGZERO/MAGZP/ZP_STACK/ZPAB keyword found in the FITS headers.\n"
                             "By default, check_zeropoints() multiplies each --scales value by\n"
                             "a correction factor derived from the header's zeropoint (if any),\n"
                             "so a header-based adjustment is applied on top of your --scales\n"
                             "even when --scales is set explicitly. This flag skips that check\n"
                             "entirely, making --scales the final word.")
    parser.add_argument("--fr", type=float, default=0.3,
                        help="H-blending fraction for L channel")
    parser.add_argument("--blendIY", action="store_true",
                        help="Blend I and Y into blue channel (B = (Y + fi*I)/(1+fi)), with J as green.\n"
                             "Default: I→blue, average(Y,J)→green.")
    parser.add_argument("--fi", type=float, default=1.6,
                        help="I-blending fraction for B channel (only with --blendIY)")
    parser.add_argument("--subtract_back", action="store_true",
                        help="Estimate and subtract a constant background from each band.\n"
                             "Background is estimated from a 5%% random pixel subsample: median\n"
                             "and MAD are computed, pixels beyond +/-1.486*MAD of the median are\n"
                             "rejected, and a Gaussian (sigma fixed to 1.486*MAD) is fit to the\n"
                             "histogram of the surviving pixels. The fitted mean is subtracted.\n"
                             "Applied immediately after reading each FITS file, before plate-scale\n"
                             "resampling and flux scaling.")

    # --- Hot-pixel masking ---
    parser.add_argument("--mask", nargs="?", default=None, const=True, type=str2bool,
                        help="Mask hot pixels; auto-applied to EWS images unless set to False")

    # --- Colour pipeline ---
    parser.add_argument("--pivot", type=float, default=0.15,
                        help="Asinh pivot parameter, in the same units as --blackwhite -- i.e.\n"
                             "the common, zero-point-independent flux scale after --scales\n"
                             "correction, not raw pixel counts (see --blackwhite for why).\n"
                             "Controls the linear-to-log transition: at flux = 1/pivot the\n"
                             "stretch transitions from linear to logarithmic.\n"
                             "Smaller = stronger compression of faint features.\n"
                             "Default 0.15 (same as the previous eummy default).")
    parser.add_argument("--saturate", nargs="+", type=float, default=[2.0],
                        help="Colour saturation factor(s).  One value: applied to both a* and b*.\n"
                             "Two values: applied to a* and b* separately.")
    parser.add_argument("--sat-threshold", type=float, default=10.0,
                        help="L* value at which saturation boost reaches half its full value.\n"
                             "Below this level (background, noise) saturation ≈ 1 (no boost).\n"
                             "Above it (galaxies, sources) saturation → --saturate value.\n"
                             "Default 10.0.  Background L* is typically 1–5 for dark fields.")
    parser.add_argument("--sat-width", type=float, default=5.0,
                        help="Width of the sigmoid saturation taper in L* units.\n"
                             "Smaller = sharper transition.  Default 5.0.")
    parser.add_argument("--contrast", type=float, default=None,
                        help="Additional S-curve contrast boost applied to the dynamic-range-\n"
                             "stretched L channel only (leaves B/G/R untouched, so hue is not\n"
                             "affected by this step). 1.0: EWS (auto), 1.6: EDS (auto), 0: off.")

    # --- CVD ---
    parser.add_argument("--cvd", nargs="?", type=float, default=None, const=1.0, metavar="K",
                        help="Apply daltonization for red-green colour-vision deficient viewers\n"
                             "(Fidaner+2005, based on Vienot+1999 simulation).\n"
                             "K in [0,1] sets correction strength (default 1.0=full).\n"
                             "Example: --cvd  or  --cvd 0.7")
    parser.add_argument("--cvd-type", default="deutan", choices=["deutan", "protan"],
                        help="deutan (default): deuteranopia/deuteranomaly (~6%% of males).\n"
                             "protan: protanopia/protanomaly (~2%% of males).")

    # --- Diagnostics ---
    parser.add_argument("--diag", action="store_true",
                        help="Accumulate a*b* chrominance pixel values into FITS bin files\n"
                             "for diagnostic histogram plots.")

    # --- Unsharp masking ---
    parser.add_argument("--UM", nargs="*", default=["1.6", "0.75", "0.09"],
                        help="Unsharp masking: FWHM strength threshold, or False to disable")

    # --- Output ---
    parser.add_argument("--output", default="TILE[id].tif", help="Output file name")
    parser.add_argument("--nthreads", type=int, default=os.cpu_count() // 2,
                        help="Number of threads for parallel operations")

    # --- Cutouts ---
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
        "  Output: 16-bit TIFF named <TILE>_<RA>+<Dec>.tif\n"
        "  Edge pixels outside the image are rendered black.\n"
    )
    parser.add_argument("--cutout", nargs="+", metavar="VAL", help=cutout_help)
    parser.add_argument("--cutouts", nargs="+", metavar="VAL",
                        help="Text file: one cutout per line, same format as --cutout arguments;\n"
                             "  lines starting with # are ignored.  Usage: --cutouts FILE\n"
                             "FITS catalog: RA/Dec columns auto-detected, radius required.\n"
                             "  Usage: --cutouts catalog.fits 30\"\n")

    args = parser.parse_args()
    args.UM = parse_um(args.UM)

    # Parse --saturate: one value → same for a* and b*; two values → (sa, sb)
    if len(args.saturate) == 1:
        args.saturate_a = args.saturate[0]
        args.saturate_b = args.saturate[0]
    elif len(args.saturate) == 2:
        args.saturate_a = args.saturate[0]
        args.saturate_b = args.saturate[1]
    else:
        parser.error("--saturate takes 1 or 2 values")

    # Reconstruct the invoked command line (shlex.join shell-quotes any
    # argument containing spaces, e.g. sexagesimal --cutout coordinates or
    # paths with spaces), so it can be embedded in the output TIFF metadata
    # for provenance -- lets you later recover exactly how an image was made.
    args.command = shlex.join(sys.argv)

    return args, parser

# ---------------------------------------------------------------------------
# File discovery and header reading
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

# Filename pattern groups to try, in order, when auto-discovering the four
# band images in a directory. Each group is (I, Y, J, H) glob patterns.
# Different pipelines producing Euclid-style MER stacks use different
# naming conventions; the first group is the original/standard Euclid MER
# BGSUB-MOSAIC convention, the other two are alternate conventions seen
# from other pipelines. Each pattern gets a trailing '*' before '.fits' if
# it doesn't already end in one, so version tags, tile IDs, dates etc.
# after the identifying substring don't prevent a match.
_MER_NAMING_GROUPS = [
    ("EUC_MER_BGSUB-MOSAIC-VIS*.fits",
     "EUC_MER_BGSUB-MOSAIC-NIR-Y*.fits",
     "EUC_MER_BGSUB-MOSAIC-NIR-J*.fits",
     "EUC_MER_BGSUB-MOSAIC-NIR-H*.fits"),
    ("EUC_VIS_LSB*.fits",
     "EUC_NISP_Y_LSB*.fits",
     "EUC_NISP_J_LSB*.fits",
     "EUC_NISP_H_LSB*.fits"),
    ("EUC_VIS_*STK*.fits",
     "EUC_NIR_*STK_Y*.fits",
     "EUC_NIR_*STK_J*.fits",
     "EUC_NIR_*STK_H*.fits"),
]

def find_images_in_directory(path, parser):
    """Locate exactly one FITS file per band (VIS, NIR-Y, NIR-J, NIR-H) in
    path, trying each entry in _MER_NAMING_GROUPS in turn until one group
    yields exactly one match for all four bands. Patterns from different
    groups are never mixed within a single match -- if a group matches
    some bands but not all four exactly once, the whole group is rejected
    and the next one is tried, rather than combining partial matches from
    different conventions.
    Exits with an error if no group matches all four bands exactly once.
    """
    for i_pat, y_pat, j_pat, h_pat in _MER_NAMING_GROUPS:
        vis_images   = glob.glob(os.path.join(path, i_pat))
        nir_y_images = glob.glob(os.path.join(path, y_pat))
        nir_j_images = glob.glob(os.path.join(path, j_pat))
        nir_h_images = glob.glob(os.path.join(path, h_pat))

        if (len(vis_images) == 1 and len(nir_y_images) == 1
                and len(nir_j_images) == 1 and len(nir_h_images) == 1):
            tileID = extract_tileID(vis_images[0])
            return vis_images[0], nir_y_images[0], nir_j_images[0], nir_h_images[0], tileID

    print(f"Error: Could not find exactly one image per band (VIS, NIR-Y, "
          f"NIR-J, NIR-H) in {path} using any recognised naming convention.\n")
    parser.print_help()
    sys.exit(1)


def get_science_hdu(hdul):
    """Return the HDU that actually contains image data.

    Most FITS files store the image in the primary HDU (index 0). Some
    pipelines instead leave the primary HDU empty (a minimal placeholder
    header, no data -- sometimes used only to hold top-level metadata) and
    put the real image, and often the full WCS/photometric header too, in
    the first extension. This returns whichever HDU actually has data,
    preferring the primary HDU when it has data, otherwise scanning
    forward through the extensions for the first one that does.

    Raises ValueError if no HDU in the file contains image data.
    """
    if hdul[0].data is not None:
        return hdul[0]
    for hdu in hdul[1:]:
        if hdu.data is not None:
            return hdu
    raise ValueError("No HDU with image data found in FITS file")


def read_zeropoint(fits_path):
    """Read the photometric zeropoint from fits_path's header, checking
    MAGZERO, MAGZP, ZP_STACK, ZPAB in that order (first one found wins).

    Checks both the primary header and the science (data-bearing) HDU's
    header, since real files fall into three cases: (1) the primary HDU
    carries both the data and the zeropoint keyword; (2) the primary HDU
    is empty of data but still carries the zeropoint (data lives in an
    extension whose own header does not repeat it); (3) both the data and
    the zeropoint live together in an extension, with a near-empty primary
    HDU. Checking only the science HDU (as get_science_hdu alone would
    give you) misses case 2; checking only the primary misses case 3.
    The science HDU's header is checked first (matches get_plate_scale/
    extract_wcs's convention of treating it as authoritative when both
    are present); the primary header is consulted only as a fallback.

    Returns None if none of these keywords are present in either header.
    """
    with fits.open(fits_path) as hdul:
        science_hdr = get_science_hdu(hdul).header
        primary_hdr = hdul[0].header
        for hdr in (science_hdr, primary_hdr):
            for kw in ('MAGZERO', 'MAGZP', 'ZP_STACK', 'ZPAB'):
                if kw in hdr:
                    return float(hdr[kw])
    return None

def check_zeropoints(i_band, y_band, j_band, h_band, args):
    """Read MAGZERO (or MAGZP, ZP_STACK, ZPAB) from each FITS header and
    compare to the reference zeropoints for which the default --scales
    values are calibrated:

        VIS (I): 24.5   Y: 29.8   J: 30.1   H: 30.0

    If the measured zeropoint differs from the reference, the corresponding
    scale factor is corrected by 10^((zp_measured - zp_ref) / 2.5).
    If no zeropoint keyword is found, the default scale is kept unchanged.
    """
    ZP_REF = {'I': 24.5, 'Y': 29.8, 'J': 30.1, 'H': 30.0}

    def abs_path(f):
        return f if os.path.isabs(f) else os.path.join(args.path, f)

    bands = [('I', abs_path(i_band)), ('Y', abs_path(y_band)),
             ('J', abs_path(j_band)), ('H', abs_path(h_band))]

    si, sy, sj, sh = args.scales
    scale_map = {'I': si, 'Y': sy, 'J': sj, 'H': sh}

    print("Checking photometric zero points")

    for band, fpath in bands:
        zp_ref = ZP_REF[band]
        zp_measured = read_zeropoint(fpath)
        if zp_measured is None:
            continue
        delta = zp_measured - zp_ref
        if abs(delta) > 0.01:
            correction = 10.0 ** (delta / 2.5)
            scale_map[band] = scale_map[band] * correction

    args.scales = [scale_map['I'], scale_map['Y'], scale_map['J'], scale_map['H']]

def get_plate_scale(fits_path):
    """Return the pixel scale of fits_path in arcsec/pixel, read from its WCS.

    Uses astropy.wcs.WCS.proj_plane_pixel_scales(), which handles both the
    CD-matrix and CDELT+PC formalisms transparently. Returns the mean of the
    x/y pixel scales; warns if they differ by more than 0.1% (non-square
    pixels), which would indicate a distorted or unusual WCS.

    Returns None (with a warning) if the header has no valid celestial WCS
    -- e.g. missing CTYPE1/CTYPE2, which some crop/repackage tools (SWarp,
    custom fitscut-style utilities) silently drop or reset to a dummy
    linear placeholder (CDELT=1, CRVAL=0). Without CTYPE, astropy cannot
    tell the axes are angular, and proj_plane_pixel_scales() returns
    dimensionless values that cannot be converted to arcsec -- this is
    caught here rather than left to crash with an opaque UnitConversionError
    deep in astropy's unit-conversion machinery.
    """
    with fits.open(fits_path) as hdul:
        header = get_science_hdu(hdul).header
        wcs = WCS(header)

    if not wcs.has_celestial:
        print(f"  WARNING: {os.path.basename(fits_path)} has no valid celestial WCS "
              f"(missing or invalid CTYPE1/CTYPE2) — cannot determine its plate scale. "
              f"This often means an upstream cropping/repackaging step stripped or "
              f"reset the WCS. Proceeding as if this band's plate scale matches the "
              f"reference (no resampling for this band) — verify this is correct.")
        return None

    scales_arcsec = [s.to(u.arcsec).value for s in wcs.proj_plane_pixel_scales()]
    if abs(scales_arcsec[0] - scales_arcsec[1]) / scales_arcsec[0] > 1e-3:
        print(f"  WARNING: non-square pixels in {os.path.basename(fits_path)}: "
              f"{scales_arcsec[0]:.5f}\" x {scales_arcsec[1]:.5f}\" — using mean")
    return float(np.mean(scales_arcsec))

def extract_wcs(fits_path):
    """Read the WCS of fits_path and return it as a plain dict, in the
    CD-matrix formalism (CD1_1 ... CD2_2), which encodes pixel scale,
    orientation, and shear in a single linear transform.

    Uses astropy.wcs.WCS.pixel_scale_matrix rather than reading CD1_1...
    CD2_2 directly from the header: many pipelines express the same
    linear WCS via CDELT1/2 + PC1_1... instead of an explicit CD matrix,
    and reading the CD keywords directly would silently return None for
    such headers, producing a TIFF with broken WCS metadata. Both
    formalisms are numerically equivalent for a linear WCS (no SIP
    distortion), so normalising through the WCS object handles either.

    If the header has no valid celestial WCS (missing/invalid CTYPE1/2 --
    see get_plate_scale for common causes), the CD fields are returned as
    None rather than whatever placeholder pixel_scale_matrix would compute
    (e.g. a literal 1 deg/pixel from a CDELT=1 dummy header) -- embedding
    that into the output TIFF would silently claim real, if nonsensical,
    astrometry instead of plainly showing no WCS is available.
    """
    with fits.open(fits_path) as hdul:
        header = get_science_hdu(hdul).header
        w = WCS(header)
        wcs = {k: header.get(k) for k in ['EQUINOX', 'RADESYS', 'CTYPE1', 'CTYPE2',
                                           'CUNIT1', 'CUNIT2', 'CRVAL1', 'CRVAL2',
                                           'CRPIX1', 'CRPIX2']}
        if w.has_celestial:
            cd = w.pixel_scale_matrix
            wcs['CD1_1'], wcs['CD1_2'] = float(cd[0, 0]), float(cd[0, 1])
            wcs['CD2_1'], wcs['CD2_2'] = float(cd[1, 0]), float(cd[1, 1])
        else:
            wcs['CD1_1'] = wcs['CD1_2'] = wcs['CD2_1'] = wcs['CD2_2'] = None
    return wcs

def write_output(rgb, wcs, args):
    """Write the final uint16 RGB image as an uncompressed TIFF with the
    WCS metadata (plus the invoked command line, under COMMAND, for
    provenance) embedded, and update a 'link.tif' symlink to point to
    the new file (atomic replace to avoid a broken-link window)."""
    output_path = os.path.join(args.path, f"{args.output}")

    metadata = dict(wcs)
    metadata['COMMAND'] = args.command

    print(f"Writing result to {output_path}")
    tifffile.imwrite(
        output_path,
        rgb,
        metadata=metadata,
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
# Cutout specification parsing (coordinates, dimensions, catalogs)
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
        w = WCS(get_science_hdu(hdul).header)
    ps = w.proj_plane_pixel_scales()   # returns Quantity array
    pix_scale_arcsec = (ps[0].to(u.arcsec).value + ps[1].to(u.arcsec).value) / 2

    arcsec = val * 60.0 if unit == 'arcmin' else val
    return max(1, int(round(arcsec / pix_scale_arcsec)))

def _sky_to_pixel(c1_str, c2_str, fits_path):
    """Convert sky coordinates (sexagesimal or decimal degrees) to
    0-based pixel (col, row) using the FITS WCS.  Returns (col, row)."""
    with fits.open(fits_path) as hdul:
        w = WCS(get_science_hdu(hdul).header)

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
        w = WCS(get_science_hdu(hdul).header)
    coord = w.pixel_to_world(col, row)
    return float(coord.ra.deg), float(coord.dec.deg)

def _cutout_filename(base_tif, ra_deg, dec_deg):
    """Build the TIFF filename from the base name and sky position.
    e.g. TILE12345678_100.345655-47.567834.tif
    Precision: 6 decimal degrees ≈ 0.036 arcsec, well below the 0.1" requirement."""
    stem = os.path.splitext(os.path.basename(base_tif))[0]
    sign = '+' if dec_deg >= 0 else ''
    return f"{stem}_{ra_deg:.6f}{sign}{dec_deg:.6f}.tif"

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
