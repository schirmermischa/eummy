#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eummy_colorspace.py -- Lab colour-space construction and everything
downstream: the RGB->XYZ->Lab->XYZ->RGB pipeline, sRGB gamma-encoding,
unsharp masking, diagnostic a*/b* histograms, and cutout rendering.

render_lab_pipeline() is the single shared core of the Lab math -- both
colorise_image (full mosaic) and _render_cutout_crop (small crops) call
it, so the two paths behave identically by construction.

Depends on eummy_fluxspace (stretch_and_normalise_channel, contrast_adjustment,
_crop_channel) and eummy_io (write_output, cutout-spec helpers).
"""

import os, gc, cv2
import numpy as np
import numexpr as ne
import tifffile

from .eummy_fluxspace import stretch_and_normalise_channel, contrast_adjustment, _crop_channel
from .eummy_io import (write_output, _parse_cutout_tokens, _dim_to_pixels,
                        _sky_to_pixel, _pixel_to_sky, _cutout_filename)


def srgb_gamma_encode(ch):
    """Apply the sRGB transfer function (gamma-encoding) in-place,
    sign-preserving (the reconstructed linear RGB from the Lab round-trip
    can be slightly negative for out-of-gamut colours, before the final
    clip). This is the standard linear-light -> display-referred encoding,
    applied once, at the very end, to the final reconstructed RGB --
    independent of the asinh dynamic-range compression, which happens
    earlier, on L, before Lab construction.
    """
    thresh = np.float32(0.0031308)
    k  = np.float32(12.92)
    c1 = np.float32(1.055)
    c2 = np.float32(0.055)
    e  = np.float32(1.0 / 2.4)
    ne.evaluate("where(ch >= 0,"
                "  where(ch <= thresh, k*ch, c1*ch**e - c2),"
                "  where(-ch <= thresh, k*ch, -(c1*(-ch)**e - c2)))",
                local_dict={'ch': ch, 'thresh': thresh,
                            'k': k, 'c1': c1, 'c2': c2, 'e': e}, out=ch)

def render_lab_pipeline(B, G, R, L, args, nan_mask=None):
    """Core Lab colour pipeline: run the full B/G/R/L -> Lab -> RGB round
    trip on a set of same-shaped channel arrays, returning the final
    display-ready uint16 RGB array (flipped N-up, [0,65535]).

    Shared by colorise_image (full mosaic) and _render_cutout_crop
    (small crops) -- this is the ONE place the Lab math lives, so both
    paths behave identically by construction rather than via two
    hand-kept-in-sync copies.

    B, G, R arrive independently asinh-stretched to [0,1] (each stretched
    the same way as L, in main(), before this function is ever called).
    L arrives as RAW linear flux (unstretched, unnormalised).

    Pipeline:
      1. L: asinh dynamic-range stretch with black/white normalisation
         built in, then optional contrast boost, then *100 to become the
         Lab lightness scale -- done FIRST, entirely before any
         RGB->XYZ->Lab work begins. This L directly replaces the L* that
         XYZ->Lab would otherwise have produced from linear L.
      2. sRGB gamma-decode B/G/R before the XYZ matrix multiply. B/G/R
         arrived independently nonlinearly stretched (same as L); the
         historical eummy_21_/eummy_stable pipeline always treated that
         stretched image as if it were standard gamma-encoded sRGB and
         linearised it before the Lab math (matching what
         cv2.cvtColor(RGB2Lab) does internally). Skipping this step would
         leave every "linear" R/G/B value fed into the XYZ matrix far too
         large (x^2.4 for x in [0,1] is a strong compression -- e.g.
         0.5 -> 0.19 -- so omitting it inflates brightness system-wide),
         which is why colours come out far too bright and pale without it.
      3. Linear RGB → XYZ → Lab -- a*, b* from B/G/R
      4. Scale a*, b* by saturation factors; clamp to valid Lab range
      5. Replace Lab L* with the stretched L from step 1
      6. Lab → XYZ → linear RGB
      7. sRGB gamma-encode the final RGB (dynamic-range compression
         already happened in step 1; this is display encoding only)
      8. Optional unsharp masking
      9. Clamp to [0,1], scale to uint16, flip N-up
      10. Zero out nan_mask pixels (no-data), if given

    nan_mask, if given, must be the same H×W shape as B/G/R/L (FITS row
    order, not yet flipped) -- True marks no-data pixels to render black.

    Returns rgb_u16, shape (H, W, 3), uint16, RGB channel order, N-up.
    """
    # Pack colour channels into H×W×3 contiguous array; free originals
    rgb = np.empty((R.shape[0], R.shape[1], 3), dtype=np.float32)
    rgb[:, :, 0] = R;  R = None
    rgb[:, :, 1] = G;  G = None
    rgb[:, :, 2] = B;  B = None
    gc.collect()

    print("Color-space operations")

    # --- L: dynamic-range stretch, then contrast, then scale to Lab's
    # lightness range. Done FIRST, entirely before any RGB->XYZ->Lab work
    # begins -- L and B/G/R are independent until `light` is substituted
    # into Lab below. This replaces the CIE f_CIE-derived L* entirely --
    # the asinh stretch IS the perceptual-lightness-defining transform for
    # this pipeline; L* is not separately derived from linear L via f_CIE.
    print(f"Dynamic-range compression: asinh (pivot {args.pivot})")
    stretch_and_normalise_channel(L, args)
    contrast_adjustment(L, args)
    light = ne.evaluate("L * 100.0", local_dict={'L': L})
    L = None; gc.collect()

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    # --- sRGB gamma-decode B/G/R before the XYZ matrix multiply.
    # B/G/R are nonlinearly stretched (see docstring point 2), so must be
    # linearised first, exactly as the historical eummy_21_/eummy_stable
    # pipeline did.
    ne.evaluate("where(ch > 0.04045, ((ch + 0.055) / 1.055) ** 2.4, ch / 12.92)",
                local_dict={'ch': rgb[:, :, 0]}, out=rgb[:, :, 0])
    ne.evaluate("where(ch > 0.04045, ((ch + 0.055) / 1.055) ** 2.4, ch / 12.92)",
                local_dict={'ch': rgb[:, :, 1]}, out=rgb[:, :, 1])
    ne.evaluate("where(ch > 0.04045, ((ch + 0.055) / 1.055) ** 2.4, ch / 12.92)",
                local_dict={'ch': rgb[:, :, 2]}, out=rgb[:, :, 2])

    # --- Linear RGB → XYZ (D65, sRGB primaries) ---
    X = ne.evaluate("0.4124564*r + 0.3575761*g + 0.1804375*b")
    Y = ne.evaluate("0.2126729*r + 0.7151522*g + 0.0721750*b")
    Z = ne.evaluate("0.0193339*r + 0.1191920*g + 0.9503041*b")
    # r, g, b are VIEWS into rgb (not copies) -- rgb=None alone does NOT
    # free rgb's buffer while these views are still live locals; clearing
    # them too is what actually releases it here rather than only once r/g/b
    # get reassigned to the final RGB output much later in this function.
    rgb = None; r = g = b = None; gc.collect()

    # --- XYZ → Lab (f_CIE, no gamma) ---
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    ne.evaluate("X / Xn", out=X)
    ne.evaluate("Y / Yn", out=Y)
    ne.evaluate("Z / Zn", out=Z)

    eps   = np.float32(0.008856)   # (6/29)³
    kappa = np.float32(7.7870)     # 1/3·(29/6)²
    ne.evaluate("where(X > eps, X**(1.0/3.0), kappa*X + 0.137931)",
                local_dict={'X': X, 'eps': eps, 'kappa': kappa}, out=X)
    ne.evaluate("where(Y > eps, Y**(1.0/3.0), kappa*Y + 0.137931)",
                local_dict={'Y': Y, 'eps': eps, 'kappa': kappa}, out=Y)
    ne.evaluate("where(Z > eps, Z**(1.0/3.0), kappa*Z + 0.137931)",
                local_dict={'Z': Z, 'eps': eps, 'kappa': kappa}, out=Z)

    # a*, b* — raw chrominance, saturation boost applied later via sigmoid
    a_sat = ne.evaluate("500.0*(X - Y)", local_dict={'X': X, 'Y': Y})
    b_sat = ne.evaluate("200.0*(Y - Z)", local_dict={'Y': Y, 'Z': Z})

    # --- Sigmoid saturation scaling ---
    # The saturation factor is tapered smoothly as a function of L* to suppress
    # colour noise in the background (low L*) while preserving full colour
    # saturation in sources (high L*).
    #
    # w(L*) = 1 / (1 + exp(-(L* - thresh) / width))
    # effective_sat = 1 + (sat - 1) * w(L*)
    #
    # w → 0 for L* << thresh  →  effective_sat ≈ 1  (no boost, grey background)
    # w → 1 for L* >> thresh  →  effective_sat = sat (full boost, coloured sources)
    thresh = np.float32(args.sat_threshold)
    width  = np.float32(args.sat_width)
    sa     = np.float32(args.saturate_a)
    sb     = np.float32(args.saturate_b)
    ne.evaluate("a_sat * (1.0 + (sa - 1.0) / (1.0 + exp(-(light - thresh) / width)))",
                local_dict={'a_sat': a_sat, 'sa': sa,
                            'light': light, 'thresh': thresh, 'width': width},
                out=a_sat)
    ne.evaluate("b_sat * (1.0 + (sb - 1.0) / (1.0 + exp(-(light - thresh) / width)))",
                local_dict={'b_sat': b_sat, 'sb': sb,
                            'light': light, 'thresh': thresh, 'width': width},
                out=b_sat)
    # Re-clamp after saturation boost
    ne.evaluate("where(a_sat >  127,  127, where(a_sat < -128, -128, a_sat))",
                local_dict={'a_sat': a_sat}, out=a_sat)
    ne.evaluate("where(b_sat >  127,  127, where(b_sat < -128, -128, b_sat))",
                local_dict={'b_sat': b_sat}, out=b_sat)

    # Diagnostic chrominance histogram accumulation
    if args.diag:
        plot_ab_histogram(light, a_sat, b_sat, args)

    # --- Lab → XYZ → linear RGB ---
    eps_inv = np.float32(0.206896)   # 6/29
    ne.evaluate("(light + 16.0) / 116.0", local_dict={'light': light}, out=Y)
    ne.evaluate("Y + a_sat / 500.0",      local_dict={'Y': Y, 'a_sat': a_sat}, out=X)
    ne.evaluate("Y - b_sat / 200.0",      local_dict={'Y': Y, 'b_sat': b_sat}, out=Z)
    light = a_sat = b_sat = None; gc.collect()

    ne.evaluate("where(Y > eps_inv, Y**3, (Y - 0.137931)/kappa)",
                local_dict={'Y': Y, 'eps_inv': eps_inv, 'kappa': kappa}, out=Y)
    ne.evaluate("where(X > eps_inv, X**3, (X - 0.137931)/kappa)",
                local_dict={'X': X, 'eps_inv': eps_inv, 'kappa': kappa}, out=X)
    ne.evaluate("where(Z > eps_inv, Z**3, (Z - 0.137931)/kappa)",
                local_dict={'Z': Z, 'eps_inv': eps_inv, 'kappa': kappa}, out=Z)

    ne.evaluate("X * Xn", out=X)
    ne.evaluate("Y * Yn", out=Y)
    ne.evaluate("Z * Zn", out=Z)

    r = ne.evaluate("3.2404542*X - 1.5371385*Y - 0.4985314*Z")
    g = ne.evaluate("-0.9692660*X + 1.8760108*Y + 0.0415560*Z")
    b = ne.evaluate("0.0556434*X - 0.2040259*Y + 1.0572252*Z")
    X = Y = Z = None; gc.collect()

    rgb_out = np.stack([r, g, b], axis=-1)
    r = g = b = None; gc.collect()

    # sRGB gamma-encode for display (dynamic-range compression already
    # happened on L, before Lab construction -- this is display encoding only)
    for i in range(3):
        srgb_gamma_encode(rgb_out[:, :, i])

    # --- Unsharp masking ---
    if args.UM is not None:
        fwhm, strength, threshold = args.UM
        unsharp_mask(rgb_out, fwhm, strength, threshold)

    # --- Clamp to [0,1], scale to uint16, flip N-up ---
    np.clip(rgb_out, 0.0, 1.0, out=rgb_out)
    rgb_u16 = np.empty(rgb_out.shape, dtype=np.uint16)
    np.multiply(rgb_out[::-1], 65535, out=rgb_u16, casting='unsafe')
    rgb_out = None; gc.collect()

    # Zero out NaN pixels (no-data) in the final uint16 array.
    # nan_mask is in FITS row order; flip to match the N-up rgb_u16.
    if nan_mask is not None:
        rgb_u16[nan_mask[::-1]] = 0

    return rgb_u16

def colorise_image(B, G, R, L, wcs, args, parser, nan_mask=None):
    """Render the full mosaic through the Lab pipeline and write it to disk.
    See render_lab_pipeline() for the full colour-pipeline description --
    this is a thin wrapper adding only the file-write step, so the cutout
    path (_render_cutout_crop) can share the exact same rendering code."""
    rgb_u16 = render_lab_pipeline(B, G, R, L, args, nan_mask=nan_mask)
    write_output(rgb_u16, wcs, args)
    rgb_u16 = None; gc.collect()

# ---------------------------------------------------------------------------
# Post-processing and diagnostics
# ---------------------------------------------------------------------------

def unsharp_mask(image, radius=1.6, strength=0.75, threshold=0.09,
                 clip_min=0.0, clip_max=1.0):
    """Sharpen image by subtracting a Gaussian-blurred version of itself.

    Only pixels where the local contrast (|image - blurred|) exceeds
    threshold are sharpened, which avoids amplifying noise in flat regions.
    The result is clipped to [clip_min, clip_max] in-place.  For RGB images
    the default [0, 1] range applies; for Lab channels, pass appropriate
    bounds (e.g. [0, 100] for L*, [-128, 127] for a*/b*).

    All operations are fused into single numexpr passes to avoid intermediate
    allocations.
    """
    print(f"Unsharp masking with {radius, strength, threshold}")
    ksize = max(3, int(2*round(radius*2.5)+1))
    blurred = cv2.GaussianBlur(image, (ksize, ksize), radius)

    ne.evaluate("where(abs(image - blurred) >= threshold, image + strength * (image - blurred), image)",
                local_dict={'image': image, 'blurred': blurred, 'threshold': threshold, 'strength': strength},
                out=image)

    lo = clip_min
    hi = clip_max
    ne.evaluate("where(image > hi, hi, where(image < lo, lo, image))",
                local_dict={'image': image, 'lo': lo, 'hi': hi}, out=image)

def plot_ab_histogram(L_star, a_star, b_star, args):
    """Append per-pixel a* and b* values into shared FITS accumulator files,
    one file per L* luminosity bin, using flock-based locking for safe
    parallel execution across multiple eummy instances.

    The three bin files are written next to the calling script (or CWD):
        ab_bin1.fits   L* in [28, 40)
        ab_bin2.fits   L* in [40, 60)
        ab_bin3.fits   L* in [60, 80]

    Each file holds two columns: A_STAR and B_STAR (float32).
    Rows are appended atomically; a lock file (<bin>.lock) serialises
    concurrent writers so no data is lost.

    When --diag is given without the accumulator files being present they
    are created on first write.  The caller is responsible for running
    plot_ab_combined.py once all tiles have been processed.
    """
    import fcntl
    from astropy.io import fits as _fits
    from astropy.table import Table, vstack

    print("Accumulating a*b* pixel values into bin files")

    bins = [
        ((L_star >= 28) & (L_star < 40), "ab_bin1.fits"),
        ((L_star >= 40) & (L_star < 60), "ab_bin2.fits"),
        ((L_star >= 60) & (L_star <= 80), "ab_bin3.fits"),
    ]

    out_dir = os.getcwd()

    for mask, fname in bins:
        a_vals = a_star[mask].astype(np.float32)
        b_vals = b_star[mask].astype(np.float32)
        if a_vals.size == 0:
            continue

        fits_path = os.path.join(out_dir, fname)
        lock_path = fits_path + ".lock"

        # Open (or create) the lock file and acquire an exclusive lock.
        # The lock is held for the entire read-modify-write cycle.
        with open(lock_path, 'w') as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                new_rows = Table({'A_STAR': a_vals, 'B_STAR': b_vals})

                if os.path.exists(fits_path):
                    with _fits.open(fits_path, memmap=False) as hdul:
                        existing = Table(hdul[1].data)
                    combined = vstack([existing, new_rows])
                else:
                    combined = new_rows

                combined.write(fits_path, format='fits', overwrite=True)
                print(f"  {fname}: {len(combined):,} rows total "
                      f"(+{len(new_rows):,} from this tile)")
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

# ---------------------------------------------------------------------------
# Cutout rendering
# ---------------------------------------------------------------------------

def _render_cutout_crop(B, G, R, L, col_c, row_c, w_px, h_px, args, nan_mask=None):
    """Extract and fully render a single cutout from the four float32 channels.

    The colour pipeline is shared with the full mosaic via
    render_lab_pipeline() -- applied only to the small crop, keeping peak
    memory proportional to the cutout size rather than the full image.

    nan_mask, if given, is cropped/padded the same way as B/G/R/L, so
    cutouts get the same no-data-pixel handling as the full mosaic. Padding
    fills with True (no-data): parts of the cutout window that fall
    outside the full image are treated as no-data and render black, rather
    than as raw flux=0 run through the full stretch/Lab pipeline.

    Returns (rgb_uint16, col_start, row_start) where col_start/row_start are
    the 0-based FITS coordinates of the bottom-left corner of the cutout
    window (before clamping to image bounds), needed to adjust CRPIX1/2.
    Returns (None, 0, 0) if the centre is entirely outside the image.
    """
    B_crop, dy0, dx0 = _crop_channel(B, col_c, row_c, w_px, h_px)
    if B_crop is None:
        return None, 0, 0

    # Bottom-left corner of the cutout window in full-image 0-based coords
    col_start = int(np.floor(col_c - w_px / 2.0))
    row_start = int(np.floor(row_c - h_px / 2.0))

    G_crop, _, _ = _crop_channel(G, col_c, row_c, w_px, h_px)
    R_crop, _, _ = _crop_channel(R, col_c, row_c, w_px, h_px)
    L_crop, _, _ = _crop_channel(L, col_c, row_c, w_px, h_px)
    mask_crop, _, _ = (_crop_channel(nan_mask, col_c, row_c, w_px, h_px)
                        if nan_mask is not None else (None, 0, 0))

    ch, cw = B_crop.shape

    def _pad(crop, dtype=np.float32, fill=0.0):
        # If the crop fills the full canvas no padding is needed; copy to
        # ensure the Lab functions can write in-place without aliasing the
        # original channel arrays.
        if crop.shape == (h_px, w_px):
            return crop.copy()
        canvas = np.full((h_px, w_px), fill, dtype=dtype)
        canvas[dy0:dy0+ch, dx0:dx0+cw] = crop
        return canvas

    Bc = _pad(B_crop)
    Gc = _pad(G_crop)
    Rc = _pad(R_crop)
    Lc = _pad(L_crop)
    mask_c = _pad(mask_crop, dtype=bool, fill=True) if mask_crop is not None else None

    rgb_u16 = render_lab_pipeline(Bc, Gc, Rc, Lc, args, nan_mask=mask_c)
    return rgb_u16, col_start, row_start   # RGB order, written by tifffile

def process_cutouts(B, G, R, L, cutout_specs, fits_path, wcs, args, nan_mask=None):
    """Process all cutout specifications from the four normalised [0,1] channel
    arrays, before the memory-intensive full-image Lab conversion.

    Each cutout is rendered independently through colorise_image on its
    small crop, so peak RAM is ~5.6 GB (the four channels) plus one tiny crop.
    The output TIFF has a corrected WCS with CRPIX1/2 adjusted for the crop.

    cutout_specs: list of token lists, e.g.
        [['3000', '4000', '500p'], ['12:34:56', '-47:30:00', "5'"]]
    fits_path: path to the I-band FITS file used for WCS.
    wcs: WCS dict from extract_wcs(), used as the base for cutout WCS.
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
                col_c, row_c = _sky_to_pixel(spec['c1'], spec['c2'], fits_path)
            else:
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

        # Always derive RA/Dec for the filename
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

        rgb_out, col_start, row_start = _render_cutout_crop(
            B, G, R, L, col_c, row_c, w_px, h_px, args, nan_mask=nan_mask)
        if rgb_out is None:
            print(f"  WARNING: skipping '{desc}': cutout window entirely outside image")
            continue

        # Build cutout WCS: adjust CRPIX1/2 for the crop origin.
        # CRPIX is 1-based; col_start/row_start are 0-based.
        # The cutout is flipped vertically (N-up), same as the full TIFF, so
        # the flip does not change the CRPIX relationship.
        cutout_wcs = dict(wcs)
        if cutout_wcs.get('CRPIX1') is not None:
            cutout_wcs['CRPIX1'] = wcs['CRPIX1'] - col_start
        if cutout_wcs.get('CRPIX2') is not None:
            cutout_wcs['CRPIX2'] = wcs['CRPIX2'] - row_start
        cutout_wcs['COMMAND'] = args.command

        fname    = _cutout_filename(args.output, ra_deg, dec_deg)
        out_path = os.path.join(args.path, fname)
        tifffile.imwrite(out_path, rgb_out, metadata=cutout_wcs, compression=None)
        print(f"  Cutout saved: {out_path}")
