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
import numpy as ne
import numexpr as ne
from importlib.metadata import version, PackageNotFoundError    # to pull the version number from pyproject.toml
from concurrent.futures import ThreadPoolExecutor

# Custom formatter combining RawTextHelpFormatter and ArgumentDefaultsHelpFormatter
class CustomHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass

# Parse unsharp masking argument
def parse_um(values):
    if len(values) == 1 and values[0].lower() == "false":
        return None
    if len(values) == 3:
        try:
            return [float(v) for v in values]
        except ValueError:
            raise argparse.ArgumentTypeError("UM values must be numeric.")
    raise argparse.ArgumentTypeError("UM must be 'false' or exactly 3 floats")


# Parse boolean arguments
def str2bool(val):
    if isinstance(val, bool):
        return val
    val = val.lower()
    if val in ("yes", "true", "t", "y", "1"):
        return True
    if val in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# Command-line arguments
def parse_arguments():
# Try to get version, fallback to 'unknown' if not installed yet
    try:
        current_version = version("eummy")
    except PackageNotFoundError:
        current_version = "dev"

    print(f"\n   eummy v{version("eummy")} (Mischa Schirmer)\n")
        
    parser = argparse.ArgumentParser(
        description="Creates a colour image from Euclid MER stacks.\nRunning \"eummy \" in the directory with your images is usually sufficient.\nYou can fine-tune the result with additional command-line arguments.",
        formatter_class=CustomHelpFormatter
    )
    
    # Add the version flag
    parser.add_argument('--version', action='version', version=f'%(prog)s {current_version}')

    parser.add_argument("--path", default=os.getcwd(), help="Absolute or relative path to MER stacks")
    parser.add_argument("--images", nargs=4, help="Input FITS files for bands: I Y J H (in this order), if not following the MER naming convention.")

    parser.add_argument("--blackwhite", nargs=2, type=float, default=[-1.3, 7000],
                        help="Min/max thresholds in linear images (J-band reference)")
    parser.add_argument("--pivot", type=float, default=0.15, help="Fraction of max value used as pivot for compression")
    parser.add_argument("--contrast", type=float, default=None, help="Additional contrast curve (1.0: EWS (auto), 1.6: EDS (auto), 0: off)")

    parser.add_argument("--scales", nargs=4, type=float, default=[0.002336, 0.6532, 1.0000, 1.1448],
                        help="Scaling factors for bands I, Y, J, H")

    parser.add_argument("--fr", type=float, default=0.3, help="H blending fraction for L channel")
    parser.add_argument("--fi", type=float, default=1.6, help="I blending fraction for B channel")
    parser.add_argument("--saturate", type=float, default=2.5, help="Colour saturation factor")
    parser.add_argument("--mask", nargs="?", default=None, const=True, type=str2bool,
                        help="Mask hot pixels; automatically applied to EWS images unless set to false")
    parser.add_argument("--mergeYJ", action="store_true", help="Average Y and J into green channel")

    parser.add_argument("--UM", nargs="*", default=["1.6", "0.75", "0.09"],
                        help="Unsharp masking: FWHM strength threshold, or 'false' to disable")

    parser.add_argument("--output", default="TILE[id].tif", help="Output file name")
    parser.add_argument("--nthreads", type=int, default=os.cpu_count() // 2, help="Number of threads to use for parallel operations")

    args = parser.parse_args()
    args.UM = parse_um(args.UM)
    return args, parser


def contrast_adjustment(L, args):
    """
    Adjusts the contrast of the L channel using a 3rd order polynomial.
    Uses NumExpr for thread-safe, cache-efficient computation.
    """
    if args.contrast == 0:
        return    # no contrast adjustment requested; L is unchanged

    height, width = L.shape
    # Determine default contrast based on image dimensions (EDS vs EWS)
    if args.contrast is None:
        if height == 10200 and width == 10200:
            args.contrast = 1.6
            print(f"Enhancing contrast by {args.contrast} (default for EDS)")
        else:
            args.contrast = 1.0
            print(f"Enhancing contrast by {args.contrast} (default for EWS)")
    else:
        print(f"Enhancing contrast by {args.contrast}")

    # The polynomial: y = 0.5707*x^3 - 1.8298*x**2 + 2.2592*x
    # The adjustment logic: args.contrast * (y_poly - x) + x
    c = args.contrast
   
    # Horner's method to avoid invokation of power law. numexpr does it alright, though
#    ne.evaluate("c * (L * (2.2592 + L * (-1.8298 + L * 0.5707)) - L) + L",
#                local_dict={'L': L, 'c': c}, out=L)

    ne.evaluate(
        "c * (0.5707 * L**3 - 1.8298 * L**2 + 2.2592 * L - L) + L",
        local_dict={'L': L, 'c': c}, out=L)


# Extract TILE ID
def extract_tileID(filename):
    filename = os.path.basename(filename)
    match = re.search(r'(TILE\d+)\D', filename)
    if match:
        return match.group(1) + ".tif"
    else:
        return "TILE.tif"

    
# Find images
def find_images_in_directory(path, parser):
    vis_images = glob.glob(os.path.join(path, "EUC_MER_BGSUB-MOSAIC-VIS*.fits"))
    nir_y_images = glob.glob(os.path.join(path, "EUC_MER_BGSUB-MOSAIC-NIR-Y*.fits"))
    nir_j_images = glob.glob(os.path.join(path, "EUC_MER_BGSUB-MOSAIC-NIR-J*.fits"))
    nir_h_images = glob.glob(os.path.join(path, "EUC_MER_BGSUB-MOSAIC-NIR-H*.fits"))

    if len(vis_images) != 1 or len(nir_y_images) != 1 or len(nir_j_images) != 1 or len(nir_h_images) != 1:
        print(f"Error: Expected exactly one image per band in {path}.\n")
        parser.print_help()
        sys.exit(1)

    tileID = extract_tileID(vis_images[0])
    return vis_images[0], nir_y_images[0], nir_j_images[0], nir_h_images[0], tileID


def asinh_scale_and_normalise(B, G, R, L, args):
    """
    Fuses asinh scaling and normalisation into a single pass per channel.
    Avoids a full read+write cycle over all four arrays.
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

        
def repair_bad_pixels(B, G, R, L, args):
    print("Repairing bad pixels")
    
    mask = np.empty(L.shape, dtype=bool)

    # 1. Inpaint bad NISP pixels with VIS (keeps luminosity but removes color artifacts)
    # If any NISP band is unknown, replace all pixels with L (make it greyscale)
    ne.evaluate("((B==0) | (G==0) | (R==0)) & (L!=0)", out=mask)
    ne.evaluate("where(mask, L, B)", out=B)
    ne.evaluate("where(mask, L, G)", out=G)
    ne.evaluate("where(mask, L, R)", out=R)

    # 2. Inpaint bad VIS pixels with NISP average
    avg_nir = "(B + G + R) / 3.0"
    ne.evaluate(f"where((B!=0) & (G!=0) & (R!=0) & (L==0), {avg_nir}, L)", out=L)

    # 3. Handle Hot Pixels
    # Done sequentially to avoid race conditions on shared arrays; exploiting numpy's internal vectorisation
    # This might mask objects with very strong colors, in this case set --mask false
    height, width = L.shape

    #--mask not provided -> None -> auto-applies only for large images
    #--mask (no value) -> True -> always apply
    #--mask false -> False -> never apply
    if (height > 15000 and width > 15000 and args.mask is not False) or args.mask is True:

        # thresh4 must be comparatively high because VIS PSF is very compact;
        # otherwise be stamp out the cores of compact stars
        thresh1, thresh2, thresh3, thresh4 = 5, 5, 3, 20
        
        # Identify hot pixels in each channel
        ne.evaluate("(B > th1) & (B > th2 * (G+R+L)/3)",
                    local_dict={'B': B, 'G': G, 'R': R, 'L': L, 'th1':thresh1,'th2':thresh2}, out=mask)
        ne.evaluate("where(mask, (G+R+L)/3, B)", out=B)

        ne.evaluate("(G > th1) & (G > th2 * (B+R+L)/3)",
                    local_dict={'B': B, 'G': G, 'R': R, 'L': L, 'th1':thresh1,'th2':thresh2}, out=mask)
        ne.evaluate("where(mask, (B+R+L)/3, G)", out=G)
        
        ne.evaluate("(R > th1) & (R > th2 * (B+G+L)/3)",
                    local_dict={'B': B, 'G': G, 'R': R, 'L': L, 'th1':thresh1,'th2':thresh2}, out=mask)
        ne.evaluate("where(mask, (B+G+L)/3, R)", out=R)
        
        ne.evaluate("(L > th3) & (L > th4 * (B+G+R)/3)",
                    local_dict={'B': B, 'G': G, 'R': R, 'L': L, 'th3':thresh3,'th4':thresh4}, out=mask)
        ne.evaluate("where(mask, (B+G+R)/3, L)", out=L)

    # 4. Final Saturated Pixel Handling
    # Use a high value (white) for any remaining clipped/zero pixels
    ne.evaluate("(B==0) | (G==0) | (R==0) | (L==0)", out=mask)
    B[mask] = 1e5
    G[mask] = 1e5
    R[mask] = 1e5

    
def unsharp_mask(image, radius=1.6, strength=0.75, threshold=0.09):
    print(f"Unsharp masking with {radius, strength, threshold}")
    ksize = max(3, int(2*round(radius*2.5)+1))
    blurred = cv2.GaussianBlur(image, (ksize, ksize), radius)

    # Fuse mask computation and sharpening into a single pass; no intermediate buffer needed
    ne.evaluate("where(abs(image - blurred) >= threshold, image + strength * (image - blurred), image)",
                local_dict={'image': image, 'blurred': blurred, 'threshold': threshold, 'strength': strength},
                out=image)

    # Clip in-place
    ne.evaluate("where(image > 1, 1, where(image < 0, 0, image))", out=image)

    
# WCS extraction
def extract_wcs(fits_path):
    with fits.open(fits_path) as hdul:
        header = hdul[0].header
        wcs = {k:header.get(k) for k in ['EQUINOX','RADESYS','CTYPE1','CTYPE2',
                                         'CUNIT1','CUNIT2','CRVAL1','CRVAL2',
                                         'CRPIX1','CRPIX2','CD1_1','CD1_2','CD2_1','CD2_2']}
    return wcs


def rescale_and_blend(args,parser):
    if not os.path.isdir(args.path):
        print(f"Error: Directory '{args.path}' does not exist.")
        sys.exit(1)

    # If the --images argument is provided
    if args.images:
        i_band, y_band, j_band, h_band = args.images
        tileID = extract_tileID(i_band)
        
    else:
        # look for images in the specified directory
        i_band, y_band, j_band, h_band, tileID = find_images_in_directory(args.path, parser)

    # determine the final output file name
    if args.output == "TILE[id].tif":
        # default, user did not provide anything on the command line; override with automatic value
        args.output = tileID

    # Extract individual scale factors from the args
    si, sy, sj, sh = args.scales

    print("Processing FITS images")

    def load_and_prep(file_path):
        with fits.open(file_path) as hdul:
            data = hdul[0].data
            # Returns a view if already float32, otherwise converts in-place if possible
            return np.asanyarray(data, dtype=np.float32)

    # 2. Parallel Reading (I/O Bound)
    # get file names, with a guard against the user providing absolute paths in both --path and --images
    files = [f if os.path.isabs(f) else os.path.join(args.path, f) for f in [y_band, j_band, h_band, i_band]]
    # Overlap disk-read latency for all four bands simultaneously
    with ThreadPoolExecutor(max_workers=4) as executor:
        y_data, j_data, h_data, i_data = list(executor.map(load_and_prep, files))

    # 3. Extract WCS from the VIS band (standard for Euclid MER);
    # guard against double absolute paths in --path and --images
    wcs = extract_wcs(i_band if os.path.isabs(i_band) else os.path.join(args.path, i_band))

    # 4. Individual Pre-scaling (Parallel via NumExpr)
    # We only perform the math if the scale factor is not 1.0
    
    scaling_tasks = [
        (y_data, sy, "Y-band"),
        (j_data, sj, "J-band"),
        (h_data, sh, "H-band"),
        (i_data, si, "VIS-band")
    ]

    for data, scale, name in scaling_tasks:
        if scale != 1.0:
            # ne.evaluate handles the division in-place to stay in CPU cache
            ne.evaluate("data / scale", local_dict={'data': data, 'scale': scale}, out=data)

    # Fix bad pixels
    repair_bad_pixels(y_data, j_data, h_data, i_data, args)

    # 1. Blending B channel
    fi = args.fi
    if args.mergeYJ:
        B = i_data
    else:
        # Fuses addition and division into one pass
        B = ne.evaluate("(y_data + i_data * fi) / (1.0 + fi)", 
                        local_dict={'y_data': y_data, 'i_data': i_data, 'fi': fi})

    # 2. Blending G channel
    if args.mergeYJ:
        G = ne.evaluate("(y_data + j_data) * 0.5", local_dict={'y_data': y_data, 'j_data': j_data})
    else:
        G = j_data

    # 3. Blending L (Luminance) channel
    fr = args.fr
    if fr > 0:
        L = ne.evaluate("(i_data + fr * exp(-0.2*abs(i_data)) * h_data) / (1.0 + fr * exp(-0.2*abs(i_data)))",
                        local_dict={'i_data': i_data, 'h_data': h_data, 'fr': fr})
    else:
        L = i_data

    # R-band
    R = h_data

    # Free temporaries that are no longer aliased by B, G, R, L
    y_data = None               # consumed into B
    if fr > 0: i_data = None    # consumed into L; if fr==0, L=i_data so keep it
    gc.collect()

    return B, G, R, L, wcs


def write_output(rgb, wcs, args):
    output_path = os.path.join(args.path, f"{args.output}")

    print(f"Writing result to {output_path}")
    tifffile.imwrite(
        output_path,
        rgb,
        metadata=wcs,
        compression=None
        # tile=(512,512),           # slows down the write on fast-I/O systems 
        # maxworkers=args.nthreads  # Use all cores for the tiling
    )

    # Free the input buffer
    rgb = None
    gc.collect()

    # Update the symlink
    link_name = os.path.join(args.path, "link.tif")
    tmp_link = link_name + ".tmp"
    try:
        os.symlink(args.output, tmp_link)
        os.replace(tmp_link, link_name)
    except Exception as e:
        if os.path.exists(tmp_link):
            os.unlink(tmp_link)
        raise

def rgb_lab_rgb_OpenCV(rgb, L, args):
    print("Color-space operations")
    # Convert RGB to CIELab
    # Lab space: [0] = Lightness, [1] = a (green-red), [2] = b (blue-yellow)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    rgb = None; gc.collect()

    # 1. Scale saturation in-place 
    s = args.saturate
    for i in [1, 2]:  # Process 'a' and 'b' channels
        ch_view = lab[:, :, i]
        # Calculate saturation and clip to valid Lab range [-128, 127] in one pass
        ne.evaluate("where(ch*s > 127, 127, where(ch*s < -128, -128, ch*s))", local_dict={'ch': ch_view, 's': s},
                    out=ch_view)

    # 2. Assign the L (Luminance) channel
    # Lab Lightness is 0-100, so we stretch the L [0,1] section accordingly
    # inefficient (creates temporary copy)
    # lab[:, :, 0] = L * 100
    # efficient: do it in Lab directly
    ne.evaluate("L * 100.0", local_dict={'L': L}, out=lab[:, :, 0])
    L = None; gc.collect()

    # 3. Convert back to RGB
    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    lab = None
    gc.collect()

    return rgb


# Much 5x slower than the openCV implementation, because the latter uses a LUT for the power-law in the gamma function 
# UNUSED. Shown for completeness, only
def rgb_lab_rgb_manual(rgb, L, args):
    print("Color-space operations")
    # --- RGB -> Lab ---
    # Step 1: Linear RGB to XYZ (D65, sRGB primaries matrix)
    # gamma correction (can't skip)
    ne.evaluate("where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)",
                local_dict={'rgb': rgb}, out=rgb)

    height, width = rgb.shape[:2]
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    X = ne.evaluate("0.4124564*r + 0.3575761*g + 0.1804375*b")
    Y = ne.evaluate("0.2126729*r + 0.7151522*g + 0.0721750*b")
    Z = ne.evaluate("0.0193339*r + 0.1191920*g + 0.9503041*b")
    rgb = None  # can't set to None before, because r, g, b are views into the array
    gc.collect()

    Xn = 0.95047
    Yn = 1.00000
    Zn = 1.08883
    
    # Step 2: Normalize by D65 white point
    # Xn=0.95047, Yn=1.00000, Zn=1.08883
    ne.evaluate("X / Xn", out=X)
    ne.evaluate("Y / Yn", out=Y)
    ne.evaluate("Z / Zn", out=Z)

    # Step 3: Apply CIE f-function
    # f(t) = t**(1/3) if t > 0.008856 else 7.787*t + 16/116
    eps = 0.008856    # (6/29)**3
    kappa = 7.7870     # 1/3 (29/6)**2
    ne.evaluate("where(X > eps, X ** (1.0/3.0), kappa * X + 0.137931)",
                local_dict={'X': X, 'eps': eps, 'kappa': kappa}, out=X)
    ne.evaluate("where(Y > eps, Y ** (1.0/3.0), kappa * Y + 0.137931)",
                local_dict={'Y': Y, 'eps': eps, 'kappa': kappa}, out=Y)
    ne.evaluate("where(Z > eps, Z ** (1.0/3.0), kappa * Z + 0.137931)",
                local_dict={'Z': Z, 'eps': eps, 'kappa': kappa}, out=Z)

    # Step 4: Compute chrominance and apply saturation
    s = args.saturate
    # a = 500*(fX - fY), b_ch = 200*(fY - fZ), L_lab = 116*fY - 16
    a_sat = ne.evaluate("500.0 * (X - Y) * s", local_dict={'X': X, 'Y': Y, 's': s})
    b_sat = ne.evaluate("200.0 * (Y - Z) * s", local_dict={'Y': Y, 'Z': Z, 's': s})
    # clamping to valid range
    ne.evaluate("where(a_sat >  127,  127, where(a_sat < -128, -128, a_sat))", local_dict={'a_sat': a_sat}, out=a_sat)
    ne.evaluate("where(b_sat >  127,  127, where(b_sat < -128, -128, b_sat))", local_dict={'b_sat': b_sat}, out=b_sat)

    # Use passed-in L for lightness instead of the computed one
    light = ne.evaluate("L * 100.0", local_dict={'L': L})

    # --- Lab -> RGB ---
    # Step 5: Lab -> XYZ
    ne.evaluate("(light + 16.0) / 116.0", local_dict={'light': light}, out=Y)
    ne.evaluate("Y + a_sat / 500.0", local_dict={'Y': Y, 'a_sat': a_sat}, out=X)
    ne.evaluate("Y - b_sat / 200.0", local_dict={'Y': Y, 'b_sat': b_sat}, out=Z)

    # recompute eps
    eps = 0.206896   # 6/29, no power-of-three here!
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

    # Step 6: XYZ -> linear RGB (inverse sRGB matrix)
    r = ne.evaluate("3.2404542*X - 1.5371385*Y - 0.4985314*Z")
    g = ne.evaluate("-0.9692660*X + 1.8760108*Y + 0.0415560*Z")
    b = ne.evaluate("0.0556434*X - 0.2040259*Y + 1.0572252*Z")
    X = Y = Z = a_sat = b_sat = L_lab = None
    gc.collect()

    # Step 7: Apply sRGB gamma and clip
    ne.evaluate("where(r > 0.0031308, 1.055 * r**(1.0/2.4) - 0.055, 12.92 * r)", out=r)
    ne.evaluate("where(g > 0.0031308, 1.055 * g**(1.0/2.4) - 0.055, 12.92 * g)", out=g)
    ne.evaluate("where(b > 0.0031308, 1.055 * b**(1.0/2.4) - 0.055, 12.92 * b)", out=b)

    # clamp and stack back
    ne.evaluate("where(r > 1, 1, where(r < 0, 0, r))", out=r)
    ne.evaluate("where(g > 1, 1, where(g < 0, 0, g))", out=g)
    ne.evaluate("where(b > 1, 1, where(b < 0, 0, b))", out=b)
    rgb = np.stack([r, g, b], axis=-1)
    r = g = b = None

    return rgb

def colorise_L(B, G, R, L, wcs, args, parser):
    """
    Combines the processed R, G, B channels with the L (Luminance) channel 
    using the CIELab color space. Optimized for memory and speed.
    """
    # Stack channels into a float32 RGB image;

    # don't let numpy make a copy if data is already in 32bit (which it is, but anyway)
    # waste of memory, not used
    # rgb = np.stack([R, G, B], axis=-1).astype(np.float32, copy=False)
    # better:
    rgb = np.empty((R.shape[0], R.shape[1], 3), dtype=np.float32)
    rgb[:, :, 0] = R;  R = None
    rgb[:, :, 1] = G;  G = None
    rgb[:, :, 2] = B;  B = None
    gc.collect()   # probably uneffective, but nonetheless, this function is the one with highest memory usage

    # OpenCV implementation / manual implementation
    # rgb = rgb_lab_rgb_manual(rgb, L, args)
    rgb = rgb_lab_rgb_OpenCV(rgb, L, args)
    L = None   # ← free here, rgb_lab_rgb_OpenCV is done with it
    gc.collect()
    
    # 4. Apply Unsharp Masking if enabled
    if args.UM is not None:
        fwhm, strength, threshold = args.UM
        unsharp_mask(rgb, fwhm, strength, threshold)

    # 5. Prepare for output (trying to be efficient, avoiding negative strides in memory so that tiffflib doesn't have to reorder)
    print("16-bit conversion")
    rgb_out = np.empty(rgb.shape, dtype=np.uint16)
    np.multiply(rgb[::-1], 65535, out=rgb_out, casting='unsafe')  # rgb_out is contiguous, written top-to-bottom
    rgb = None
    gc.collect()

    # 6. Be done
    write_output(rgb_out, wcs, args)


# Main function
def main():
    # Keep main simple; let the helper function handle the parser
    args, parser = parse_arguments()
    ne.set_num_threads(args.nthreads)
    cv2.setNumThreads(args.nthreads)   # probably uneffective, but just in case
    
    B,G,R,L,wcs = rescale_and_blend(args, parser)
    asinh_scale_and_normalise(B,G,R,L,args)
    contrast_adjustment(L,args)
    colorise_L(B,G,R,L,wcs,args,parser)

if __name__=="__main__":
    main()
