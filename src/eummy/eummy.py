#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# eummy.py - A program to create color images from Euclid MER stacks
# Copyright (C) 2025 Mischa Schirmer

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.

import sys, os, glob, argparse, cv2, gc, re, tifffile, psutil, numpy as np
from astropy.io import fits
from skimage import img_as_uint, img_as_ubyte
from scipy.ndimage import affine_transform
from scipy.interpolate import InterpolatedUnivariateSpline
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numexpr as ne

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
    parser = argparse.ArgumentParser(
        description="Creates a colour image from Euclid MER stacks.\nRunning \"eummy.py --path /path/to/MERstacks/\" is usually sufficient.\nYou can fine-tune the result with additional command-line arguments.",
        formatter_class=CustomHelpFormatter
    )

    parser.add_argument("--path", help="Absolute or relative path to MER stacks")
    parser.add_argument("--images", nargs=4, help="Input FITS files for bands: I Y J H (in this order), if not following the MER naming convention.")

    parser.add_argument("--blackwhite", nargs=2, type=float, default=[-2, 10000],
                        help="Min/max thresholds in linear images (J-band reference)")
    parser.add_argument("--pivot", type=float, default=0.15, help="Fraction of max value used as pivot for compression")
    parser.add_argument("--strength", default="single", help="Use single or double asinh() compression")
    parser.add_argument("--curve", type=float, default=1.0, help="Additional contrast curve (0: off, 1: strong)")

    parser.add_argument("--scales", nargs=4, type=float, default=[0.00234, 0.65, 1.00, 1.14],
                        help="Scaling factors for bands I, Y, J, H")

    parser.add_argument("--rf", type=float, default=0.3, help="Red blending fraction for L channel")
    parser.add_argument("--bf", type=float, default=0.6, help="Blue blending fraction for B channel")
    parser.add_argument("--saturation", type=float, default=2.5, help="Saturation factor")
    parser.add_argument("--mask", nargs="?", default=True, const=True, type=str2bool,
                        help="Mask hot pixels; automatically applied to EWS images unless set to false")
    parser.add_argument("--mergeYJ", action="store_true", help="Average Y and J into green channel")

    parser.add_argument("--UM", nargs="*", default=["1.6", "0.75", "0.09"],
                        help="Unsharp masking: FWHM strength threshold, or 'false' to disable")

    parser.add_argument("--output", default="TILE[id].tiff", help="Output file name")
    parser.add_argument("--nthreads", type=int, default=os.cpu_count() // 2, help="Number of threads to use for parallel operations")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    args.UM = parse_um(args.UM)
    return args, parser

# Curve adjustment
_curve_data = np.array([
    [0.0000, 0.0000],[0.0111, 0.0282],[0.0264, 0.0672],[0.0438, 0.1089],
    [0.0556, 0.1331],[0.0771, 0.1761],[0.0944, 0.2083],[0.1139, 0.2433],
    [0.1396, 0.2849],[0.1875, 0.3602],[0.2236, 0.4153],[0.2528, 0.4610],
    [0.2917, 0.5148],[0.3243, 0.5538],[0.3493, 0.5820],[0.3757, 0.6142],
    [0.3993, 0.6411],[0.4611, 0.7083],[0.4903, 0.7352],[0.5153, 0.7581],
    [0.5500, 0.7863],[0.5785, 0.8078],[0.6250, 0.8387],[0.6597, 0.8589],
    [0.6868, 0.8750],[0.7292, 0.8965],[0.7889, 0.9261],[0.8708, 0.9583],
    [0.9410, 0.9825],[1.0000, 1.0000]
])

def curve_adjustment(L, args):
    if args.curve == 0:
        return
    print(f"Enhancing contrast by {args.curve}")
    x = _curve_data[:, 0]
    y = args.curve * (_curve_data[:, 1] - x) + x
    spline = InterpolatedUnivariateSpline(x, y)

    L_flat = L.ravel()
    n = len(L_flat)
    chunk_size = n // args.nthreads

    def process_chunk(start_idx):
        end_idx = min(start_idx + chunk_size, n)
        L_flat[start_idx:end_idx] = spline(L_flat[start_idx:end_idx])

    with ThreadPoolExecutor(max_workers=args.nthreads) as executor:
        executor.map(process_chunk, range(0, n, chunk_size))

# Extract TILE ID
def extract_tileID(filename):
    filename = os.path.basename(filename)
    match = re.search(r'(TILE\d+)\D', filename)
    if match:
        return match.group(1) + ".tiff"
    else:
        return "TILE.tiff"

# Find images
def find_images_in_directory(path):
    vis_images = glob.glob(os.path.join(path, "EUC_MER_BGSUB-MOSAIC-VIS*.fits"))
    nir_y_images = glob.glob(os.path.join(path, "EUC_MER_BGSUB-MOSAIC-NIR-Y*.fits"))
    nir_j_images = glob.glob(os.path.join(path, "EUC_MER_BGSUB-MOSAIC-NIR-J*.fits"))
    nir_h_images = glob.glob(os.path.join(path, "EUC_MER_BGSUB-MOSAIC-NIR-H*.fits"))

    if len(vis_images) != 1 or len(nir_y_images) != 1 or len(nir_j_images) != 1 or len(nir_h_images) != 1:
        print(f"Error: Expected exactly one image per band in {path}.")
        sys.exit(1)

    tileID = extract_tileID(vis_images[0])
    return vis_images[0], nir_y_images[0], nir_j_images[0], nir_h_images[0], tileID

# Dynamic-range compression
def asinh_scale(B, G, R, L, args):
    print("Dynamic-range compression")
    p = args.pivot
    channels = [B, G, R, L]

    def process_array(channel):
        flat = channel.ravel()
        n = len(flat)
        chunk_size = n // args.nthreads

        def worker(start_idx):
            end_idx = min(start_idx + chunk_size, n)
            if args.strength == "single":
                flat[start_idx:end_idx] = np.arcsinh(p * flat[start_idx:end_idx])
            else:
                flat[start_idx:end_idx] = np.arcsinh(p * np.arcsinh(p * flat[start_idx:end_idx])) / p

        with ThreadPoolExecutor(max_workers=args.nthreads) as executor:
            executor.map(worker, range(0, n, chunk_size))

    for ch in channels:
        process_array(ch)

# Normalization
def normalise_channel(ch, black, scale):
    np.subtract(ch, black, out=ch)
    np.multiply(ch, scale, out=ch)
    np.clip(ch, 0, 1, out=ch)

def normalise_floats(B, G, R, L, args):
    print(f"Setting black/white points {args.blackwhite}")
    black, white = (-0.22, 8.5) if args.strength == "single" else (-0.22, 6.8)
    scale = 1.0 / (white - black)
    channels = [B, G, R, L]
    with ThreadPoolExecutor(max_workers=args.nthreads) as executor:
        executor.map(lambda ch: normalise_channel(ch, black, scale), channels)

"""
# Repair bad pixels
def repair_bad_pixels(B, G, R, L, args):
    print("Repairing bad pixels")
    maskval = 0
    thresh1, thresh2 = 10.0, 20.0

    def inpaint_nisp():
        mask = ne.evaluate("((B==0) | (G==0) | (R==0)) & (L!=0)", local_dict={'B':B,'G':G,'R':R,'L':L})
        B[mask] = G[mask] = R[mask] = L[mask]  # replace NIR channels with L

    def inpaint_vis():
        mask = ne.evaluate("(B!=0) & (G!=0) & (R!=0) & (L==0)", local_dict={'B':B,'G':G,'R':R,'L':L})
        L[mask] = ne.evaluate("(B+G+R)/3", local_dict={'B':B,'G':G,'R':R})[mask]

    def saturate():
        mask = ne.evaluate("(B==0) | (G==0) | (R==0) | (L==0)", local_dict={'B':B,'G':G,'R':R,'L':L})
        B[mask] = G[mask] = R[mask] = maskval

    # Mask hot pixels (especially for wide survey, doesn't really get invoked for deep which is very clean)
    # This might mask objects with very strong colors, in this case set --mask false

    height, width = L.shape
    if height > 15000 and width > 15000:
#    if args.mask:
        thresh1 = 10.     # minimum brightness to be considered as a hot pixel
        thresh2 = 20.     # checking that it's really a hot pixel (could mask extremely colourful objects)
        maskB = (B > thresh1) & (B > thresh2*G)
        maskG = (G > thresh1) & (G > thresh2*R)
        maskR = (R > thresh1) & (R > thresh2*G)
        maskL = (L > thresh1) & (L > thresh2*B)
        indicesB = np.where(maskB)
        indicesG = np.where(maskG)
        indicesR = np.where(maskR)
        indicesL = np.where(maskL)
        B[indicesB] = (G[indicesB] + R[indicesB]) / 2.
        G[indicesG] = (B[indicesG] + R[indicesG]) / 2.
        R[indicesR] = (B[indicesR] + G[indicesR]) / 2.
        L[indicesL] = (B[indicesL] + G[indicesL] + R[indicesL]) / 3.

    
    def hot_pixels():
        maskB = ne.evaluate("(B>th1) & (B>th2*G)", local_dict={'B':B,'G':G,'th1':thresh1,'th2':thresh2})
        maskG = ne.evaluate("(G>th1) & (G>th2*R)", local_dict={'G':G,'R':R,'th1':thresh1,'th2':thresh2})
        maskR = ne.evaluate("(R>th1) & (R>th2*G)", local_dict={'R':R,'G':G,'th1':thresh1,'th2':thresh2})
        maskL = ne.evaluate("(L>th1) & (L>th2*B)", local_dict={'L':L,'B':B,'th1':thresh1,'th2':thresh2})
        B[maskB] = (G[maskB] + R[maskB]) / 2
        G[maskG] = (B[maskG] + R[maskG]) / 2
        R[maskR] = (B[maskR] + G[maskR]) / 2
        L[maskL] = (B[maskL] + G[maskL] + R[maskL])/3

    funcs = [inpaint_nisp, inpaint_vis, saturate]
    if args.mask:
        funcs.append(hot_pixels)

    with ThreadPoolExecutor(max_workers=args.nthreads) as executor:
        futures = [executor.submit(f) for f in funcs]
        for f in futures: f.result()

"""
def repair_bad_pixels(B, G, R, L, args):
    print("Repairing bad pixels")
    # Inpaint bad NISP pixels with VIS (making them grey-scale, while still preserving luminosity)
    mask = ((B == 0) | (G == 0) | (R == 0)) & (L != 0)
    indices = np.where(mask)
    B[indices] = L[indices]
    G[indices] = L[indices]
    R[indices] = L[indices]

    # Inpaint bad VIS pixels with NISP values
    mask = (B != 0) & (G != 0) & (R != 0) & (L == 0)
    indices = np.where(mask)
    L[indices] = (B[indices] + G[indices] + R[indices])/3.  

    # Make saturated pixels white (stellar cores, etc)
    mask = (B == 0) | (G == 0) | (R == 0) | (L == 0)
    indices = np.where(mask)
    maskval = 1e5
    B[indices] = maskval
    G[indices] = maskval
    R[indices] = maskval

    # Mask hot pixels (especially for wide survey, doesn't really get invoked for deep which is very clean)
    # This might mask objects with very strong colors, in this case set --mask false

    # Get the dimensions
    height, width = L.shape
    if height > 15000 and width > 15000:
#    if args.mask:
        thresh1 = 10.     # minimum brightness to be considered as a hot pixel
        thresh2 = 20.     # checking that it's really a hot pixel (could mask extremely colourful objects)
        maskB = (B > thresh1) & (B > thresh2*G)
        maskG = (G > thresh1) & (G > thresh2*R)
        maskR = (R > thresh1) & (R > thresh2*G)
        maskL = (L > thresh1) & (L > thresh2*B)
        indicesB = np.where(maskB)
        indicesG = np.where(maskG)
        indicesR = np.where(maskR)
        indicesL = np.where(maskL)
        B[indicesB] = (G[indicesB] + R[indicesB]) / 2.
        G[indicesG] = (B[indicesG] + R[indicesG]) / 2.
        R[indicesR] = (B[indicesR] + G[indicesR]) / 2.
        L[indicesL] = (B[indicesL] + G[indicesL] + R[indicesL]) / 3.



# Unsharp mask
def unsharp_mask(image, radius=1.6, strength=0.75, threshold=0.09, n_threads=8):
    print(f"Unsharp masking with {radius, strength, threshold}")
    ksize = max(3, int(2*round(radius*2.5)+1))
    blurred = cv2.GaussianBlur(image, (ksize, ksize), radius)
    mask = image - blurred
    H = image.shape[0]
    chunk_size = H // n_threads

    def process_rows(start, end):
        mask_chunk = mask[start:end,:]
        mask_chunk *= (np.abs(mask_chunk) >= threshold)
        image_chunk = image[start:end,:]
        image_chunk += strength * mask_chunk
        np.clip(image_chunk, 0, 1, out=image_chunk)

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for i in range(n_threads):
            start = i*chunk_size
            end = (i+1)*chunk_size if i<n_threads-1 else H
            futures.append(executor.submit(process_rows, start, end))
        for f in futures: f.result()

# WCS extraction
def extract_wcs(fits_path):
    with fits.open(fits_path) as hdul:
        header = hdul[0].header
        wcs = {k:header.get(k) for k in ['EQUINOX','RADESYS','CTYPE1','CTYPE2',
                                         'CUNIT1','CUNIT2','CRVAL1','CRVAL2',
                                         'CRPIX1','CRPIX2','CD1_1','CD1_2','CD2_1','CD2_2']}
    return wcs

# Blending functions
def blend_B(args, i_data, y_data):
    return i_data if args.mergeYJ else (i_data + y_data*args.bf)/(1+args.bf)
def blend_G(args, y_data, j_data):
    return (y_data+j_data)*0.5 if args.mergeYJ else j_data
def blend_L(args, i_data, h_data):
    if args.rf>0:
        exp_factor = np.exp(-0.2*np.abs(i_data))
        return (i_data + args.rf*h_data*exp_factor)/(1.0 + args.rf*exp_factor)
    return i_data

# Rescale and blend
def rescale_and_blend(args):
    if not os.path.isdir(args.path):
        print(f"Error: Directory '{args.path}' does not exist.")
        sys.exit(1)

    if args.images:
        i_band, y_band, j_band, h_band = args.images
        tileID = extract_tileID(i_band)
    else:
        i_band, y_band, j_band, h_band, tileID = find_images_in_directory(args.path)

    if args.output == "TILE[id].tiff":
        args.output = tileID

    scale_i, scale_y, scale_j, scale_h = args.scales
    print("\nProcessing FITS images")
    i_data = fits.getdata(os.path.join(args.path,i_band))/scale_i
    y_data = fits.getdata(os.path.join(args.path,y_band))/scale_y
    j_data = fits.getdata(os.path.join(args.path,j_band))/scale_j
    h_data = fits.getdata(os.path.join(args.path,h_band))/scale_h

    wcs = extract_wcs(os.path.join(args.path,i_band))
    repair_bad_pixels(y_data, j_data, h_data, i_data, args)

    with ThreadPoolExecutor(max_workers=args.nthreads) as executor:
        future_B = executor.submit(blend_B, args, i_data, y_data)
        future_G = executor.submit(blend_G, args, y_data, j_data)
        future_R = executor.submit(lambda h: h, h_data)
        future_L = executor.submit(blend_L, args, i_data, h_data)
        B, G, R, L = future_B.result(), future_G.result(), future_R.result(), future_L.result()

    return B, G, R, L, wcs

# Colorise L channel
def colorise_L(B, G, R, L, wcs, args, parser):
    rgb = np.stack([R,G,B], axis=-1).astype(np.float32)
    R=G=B=None; gc.collect()
    print(f"Increasing colour saturation to {args.saturation}")
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    rgb=None; gc.collect()
    H = lab.shape[0]; chunk_size = H//args.nthreads

    def process_rows(start, end):
        lab[start:end,:,1:] = np.clip(lab[start:end,:,1:]*args.saturation, -128,127)
        lab[start:end,:,0] = L[start:end,:]*100

    with ThreadPoolExecutor(max_workers=args.nthreads) as executor:
        futures = []
        for i in range(args.nthreads):
            start = i*chunk_size
            end = (i+1)*chunk_size if i<args.nthreads-1 else H
            futures.append(executor.submit(process_rows,start,end))
        for f in futures: f.result()

    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB); lab=None; gc.collect()

    if args.UM is not None:
        fwhm, strength, threshold = args.UM
        unsharp_mask(rgb, fwhm, strength, threshold, args.nthreads)

    rgb = (rgb*65535).astype(np.uint16)
    rgb[:] = rgb[::-1,:,:]

    print("Writing result to ...", end='', flush=True)
    tifffile.imwrite(os.path.join(args.path,f"{args.output}"), rgb, metadata=wcs)
    print(f" {args.path}/{args.output}")

    link_name = f"{args.path}/link.tiff"
    if os.path.islink(link_name) or os.path.exists(link_name):
        os.remove(link_name)
    os.symlink(f"{args.path}/{args.output}", link_name)

# Main function
def main():
    args, parser = parse_arguments()
    B,G,R,L,wcs = rescale_and_blend(args)
    asinh_scale(B,G,R,L,args)
    normalise_floats(B,G,R,L,args)
    curve_adjustment(L,args)
    colorise_L(B,G,R,L,wcs,args,parser)

if __name__=="__main__":
    main()
