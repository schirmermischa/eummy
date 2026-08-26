#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eummy_fluxspace.py -- Turns raw per-band FITS data into stretched,
contrast-adjusted L/B/G/R channels ready for colour-space construction.

Covers: background estimation, plate-scale resampling, bad-pixel repair,
band-to-channel blending (rescale_and_blend), the L-channel asinh
dynamic-range stretch and contrast curve, applied identically to B/G/R
before Lab construction, CVD daltonization, and cutout cropping.

Depends on eummy_io (header reading during rescale_and_blend); has no
dependency on eummy_colorspace.
"""

import sys, os, gc, cv2
import numpy as np
import numexpr as ne
from astropy.io import fits
from scipy.optimize import curve_fit
from concurrent.futures import ThreadPoolExecutor

from .eummy_io import (check_zeropoints, get_plate_scale, extract_wcs,
                        extract_tileID, find_images_in_directory, get_science_hdu)


def estimate_background(data, band_name):
    """Estimate a robust constant background level for a single band.

    A 5% random subsample of pixels is drawn (without replacement) from the
    finite (non-NaN, non-inf) pixels only — real Euclid MER tiles have NaN
    no-data borders on partial-coverage tiles, and including those in the
    sample would propagate NaN through median/MAD/histogram silently
    (np.median does not skip NaN). Filtering first, rather than sampling
    then dropping NaNs, also keeps the sample size close to the requested
    5% regardless of how large the no-data border is.

    The median and MAD (median absolute deviation) of the finite subsample
    are computed, and pixels beyond +/- 1.486*MAD of the median are rejected
    (1.486*MAD is the standard scale factor that makes the MAD a consistent
    estimator of sigma for a Gaussian). A Gaussian is then fit to the
    histogram of the surviving pixels, with sigma held fixed at 1.486*MAD
    and only the amplitude and mean left free. The fitted mean is returned
    as the background estimate.
    """
    flat = data.ravel()
    finite = flat[np.isfinite(flat)]

    if finite.size == 0:
        print(f"  WARNING: {band_name}: no finite pixels found — "
              f"skipping background subtraction (background=0.0)", flush=True)
        return 0.0

    rng = np.random.default_rng()
    n_sample = max(1, int(0.05 * finite.size))
    idx = rng.choice(finite.size, size=n_sample, replace=False)
    sample = finite[idx]

    median = np.median(sample)
    mad = np.median(np.abs(sample - median))
    sigma_robust = 1.486 * mad

    clipped = sample[np.abs(sample - median) <= sigma_robust]

    counts, edges = np.histogram(clipped, bins=50)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def gauss_fixed_sigma(x, amp, mu):
        return amp * np.exp(-0.5 * ((x - mu) / sigma_robust) ** 2)

    try:
        popt, _ = curve_fit(gauss_fixed_sigma, centers, counts,
                             p0=[counts.max(), median])
        background = popt[1]
    except RuntimeError:
        print(f"  WARNING: {band_name}: Gaussian fit did not converge — "
              f"falling back to clipped median", flush=True)
        background = median

    print(f"  {band_name}: background = {background:.6f}", flush=True)
    return background

def resample_to_reference(data, zoom_factor, target_shape, band_name):
    """Flux-conservingly resample data (float32) onto target_shape using
    bilinear interpolation.

    Uses cv2.resize (INTER_LINEAR) rather than scipy.ndimage.zoom: both
    perform the same bilinear interpolation, but cv2's SIMD/multi-threaded
    implementation runs roughly 2x faster in practice. Resizing directly to
    target_shape (rather than by an approximate zoom factor) also means the
    output shape is exact by construction, with no rounding correction
    needed.

    Bilinear resampling interpolates surface brightness (the value at each
    output pixel), not flux: the number of pixels covering any fixed
    angular aperture changes by zoom_factor**2 (the ratio of plate scales
    squared), so the raw pixel sum over that aperture would scale by
    zoom_factor**2 too. Dividing by zoom_factor**2 conserves flux, i.e.
    summing counts in a fixed aperture gives the same result before and
    after resampling.

    NaN pixels (no-data borders, cosmic rays, detector gaps) are handled
    separately from the interpolation itself. Bilinear interpolation does
    not skip NaN inputs — any output pixel whose kernel touches one becomes
    NaN too, so a single bad input pixel "bleeds" into a zoom_factor-sized
    patch of output pixels, and no-data borders grow. To avoid this, NaNs
    are filled with a placeholder before interpolating the data values, and
    the NaN footprint itself is resampled separately with nearest-neighbour
    interpolation (which reproduces each input pixel's classification
    exactly, without blending) to determine which output pixels should be
    NaN.
    """
    print(f"  Resampling {band_name} onto I-band grid "
          f"(factor {zoom_factor:.4f}x, bilinear)...", flush=True)

    dsize = (target_shape[1], target_shape[0])
    nan_mask_in = np.isnan(data)

    if nan_mask_in.any():
        finite_fill = np.nan_to_num(data, nan=0.0)
        resampled = cv2.resize(finite_fill, dsize, interpolation=cv2.INTER_LINEAR)
        resampled = resampled.astype(np.float32, copy=False)

        mask_resampled = cv2.resize(nan_mask_in.astype(np.uint8), dsize,
                                     interpolation=cv2.INTER_NEAREST)
        resampled[mask_resampled.astype(bool)] = np.nan
    else:
        resampled = cv2.resize(data, dsize, interpolation=cv2.INTER_LINEAR)
        resampled = resampled.astype(np.float32, copy=False)

    resampled /= np.float32(zoom_factor ** 2)
    return resampled

# ---------------------------------------------------------------------------
# Bad-pixel repair and channel blending
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
       value (1e5) so that it renders as white rather than black in the
       final image.
    """
    print("Repairing bad pixels")

    mask = np.empty(L.shape, dtype=bool)

    # 0. Identify NaN pixels (no-data border regions in partial-coverage tiles).
    #    Replace with sentinel −1e10 so they survive the pipeline without
    #    triggering the zero checks in steps 1-4.  After step 4 has turned
    #    genuine zero pixels (saturated stars, detector gaps) white, the NaN
    #    pixels are zeroed so they render as black (true no-data).
    SENTINEL = np.float32(-1e10)
    nan_mask = np.empty(L.shape, dtype=bool)
    ne.evaluate("(B != B) | (G != G) | (R != R) | (L != L)",
                local_dict={'B': B, 'G': G, 'R': R, 'L': L}, out=nan_mask)
    for ch in (B, G, R, L):
        ne.evaluate("where(ch != ch, SENTINEL, ch)",
                    local_dict={'ch': ch, 'SENTINEL': SENTINEL}, out=ch)

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
    L[mask] = 1e5

    # 5. Zero out original NaN pixels so they render as black.
    #    Done after step 4 so saturated-star zeros are already white.
    B[nan_mask] = 0.0
    G[nan_mask] = 0.0
    R[nan_mask] = 0.0
    L[nan_mask] = 0.0

    return nan_mask

def rescale_and_blend(args, parser):
    """Load the four Euclid MER band images and blend them into the B, G, R,
    and L channels used by the downstream colour pipeline.

    Default band-to-channel mapping
    --------------------------------
    B (blue)      : I-band (VIS)
    G (green)     : average of Y and J bands
    R (red)       : H-band
    L (luminance) : I-band with a small adaptive H contribution (--fr),
                    softened by an exponential weight that prevents bright
                    H-band sources from overwhelming faint VIS structure.

    With --blendIY, the blue channel is a weighted blend of Y and I
    (controlled by --fi), and green is J alone.

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

    # Check FITS header zeropoints and correct scales if needed, unless
    # --no-zp-check was given (then --scales is used exactly as-is)
    if args.no_zp_check:
        print("Skipping zero-point check (--no-zp-check); using --scales as given")
    else:
        check_zeropoints(i_band, y_band, j_band, h_band, args)

    si, sy, sj, sh = args.scales

    files_abs = {
        'I': i_band if os.path.isabs(i_band) else os.path.join(args.path, i_band),
        'Y': y_band if os.path.isabs(y_band) else os.path.join(args.path, y_band),
        'J': j_band if os.path.isabs(j_band) else os.path.join(args.path, j_band),
        'H': h_band if os.path.isabs(h_band) else os.path.join(args.path, h_band),
    }

    # Determine each band's plate scale from its WCS. The I-band (VIS) is
    # always the reference resolution; Y/J/H are flux-conservingly resampled
    # onto the I-band pixel grid below if their plate scale differs from it
    # by more than 0.1%. Silent unless a band actually needs resampling.
    plate_scales = {band: get_plate_scale(path) for band, path in files_abs.items()}
    ps_i = plate_scales['I']
    zoom_factors = {'I': 1.0}
    for band in ('Y', 'J', 'H'):
        ratio = plate_scales[band] / ps_i
        if abs(ratio - 1.0) > 1e-3:
            zoom_factors[band] = ratio
            print(f"  {band}: {plate_scales[band]:.5f}\"/px vs I: {ps_i:.5f}\"/px "
                  f"→ resampling by factor {ratio:.6f}")
        else:
            zoom_factors[band] = 1.0

    print("Processing FITS images", flush=True)

    def load_raw(band):
        # astropy.io.fits preserves FITS row order (row 0 = bottom of sky).
        # np.asanyarray returns a view when the data is already float32.
        with fits.open(files_abs[band]) as hdul:
            data = get_science_hdu(hdul).data
            return np.asanyarray(data, dtype=np.float32)

    # Read all four bands in parallel to overlap disk latency
    with ThreadPoolExecutor(max_workers=4) as executor:
        raw = dict(zip(['Y', 'J', 'H', 'I'], executor.map(load_raw, ['Y', 'J', 'H', 'I'])))

    # Background subtraction: applied immediately after reading the FITS
    # files, before plate-scale resampling and before flux scaling.
    if args.subtract_back:
        print("Estimating and subtracting background", flush=True)
        for band in ('I', 'Y', 'J', 'H'):
            bg = estimate_background(raw[band], band)
            data = raw[band]
            ne.evaluate("data - bg", local_dict={'data': data, 'bg': bg}, out=data)

    i_data = raw['I']
    target_shape = i_data.shape

    # Resample Y/J/H onto the I-band grid where needed. Done after loading
    # (so shapes are matched for the elementwise ops below) and before flux
    # scaling (order doesn't matter — both are linear operations).
    #
    # Bands are resampled concurrently: cv2.resize releases the GIL during
    # its C-level computation, so threading across bands can use multiple
    # cores. Each call is given a reduced internal thread count to avoid
    # oversubscribing the CPU (N concurrent calls x M threads each would
    # otherwise compete for the same cores).
    bands_to_resample = [b for b in ('Y', 'J', 'H') if zoom_factors[b] != 1.0]
    if bands_to_resample:
        threads_per_call = max(1, args.nthreads // len(bands_to_resample))
        cv2.setNumThreads(threads_per_call)
        with ThreadPoolExecutor(max_workers=len(bands_to_resample)) as executor:
            results = executor.map(
                lambda b: resample_to_reference(raw[b], zoom_factors[b], target_shape, b),
                bands_to_resample)
            raw.update(zip(bands_to_resample, results))
        cv2.setNumThreads(args.nthreads)

    y_data, j_data, h_data = raw['Y'], raw['J'], raw['H']

    # Extract and store WCS from the VIS (I) band header for use in output
    fits_i_path = files_abs['I']
    wcs = extract_wcs(fits_i_path)

    # Apply per-band flux scaling in-place (skip bands where scale == 1.0)
    for data, scale, name in [(y_data, sy, "Y"), (j_data, sj, "J"),
                               (h_data, sh, "H"), (i_data, si, "VIS")]:
        if scale != 1.0:
            ne.evaluate("data / scale", local_dict={'data': data, 'scale': scale}, out=data)

    nan_mask = repair_bad_pixels(y_data, j_data, h_data, i_data, args)

    # B (blue) and G (green) channel construction.
    # Default: B = I (VIS), G = average(Y, J).  This enhances red contrast
    # between J and H, and places VIS structure directly in the blue channel.
    # With --blendIY: B = (Y + fi*I)/(1+fi), G = J.
    fi = args.fi
    if args.blendIY:
        B = ne.evaluate("(y_data + i_data * fi) / (1.0 + fi)",
                        local_dict={'y_data': y_data, 'i_data': i_data, 'fi': fi})
        G = j_data
    else:
        B = i_data
        G = ne.evaluate("(y_data + j_data) * 0.5", local_dict={'y_data': y_data, 'j_data': j_data})

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

    return B, G, R, L, wcs, fits_i_path, nan_mask

# ---------------------------------------------------------------------------
# Dynamic-range stretch and contrast
# ---------------------------------------------------------------------------

def stretch_and_normalise_channel(ch, args):
    """Apply the asinh dynamic-range stretch to a raw linear channel, with
    black/white-point normalisation built into the stretch itself. Output
    is in [0,1] (not clamped).

    Used for L unconditionally (it always needs to become the Lab
    lightness channel), and for each of B/G/R independently, before ever
    reaching the RGB->XYZ->Lab step -- reproducing the historical
    eummy_21_/eummy_stable per-channel nonlinear-stretch behaviour.

    Stretch first, then normalise: S(x) = arcsinh(pivot*x), applied
    directly to raw flux, then black/white normalisation using the
    stretched endpoints:
        out = (S(raw) - S(vmin)) / (S(vmax) - S(vmin))
    arcsinh has no built-in anchoring (S(vmax) is not automatically 1), so
    this explicit second step is needed -- this is the historical
    eummy_21_/eummy_stable asinh_scale_and_normalise formula.

    When used for L, the output is destined to be substituted as the Lab
    lightness channel (after an optional contrast boost and a *100 scale)
    -- not written out directly -- so no clamping and no sRGB
    gamma-encoding happens here. Gamma encoding belongs on the final
    reconstructed RGB, at the very end of the pipeline.

    Operates in-place on ch.
    """
    vmin = np.float32(args.blackwhite[0])
    vmax = np.float32(args.blackwhite[1])
    pivot = np.float32(args.pivot)
    s_vmin = np.float32(np.arcsinh(float(pivot) * float(vmin)))
    s_vmax = np.float32(np.arcsinh(float(pivot) * float(vmax)))
    inv_norm = np.float32(1.0 / (s_vmax - s_vmin))
    ne.evaluate("(arcsinh(pivot * ch) - s_vmin) * inv_norm",
                local_dict={'ch': ch, 'pivot': pivot,
                            's_vmin': s_vmin, 'inv_norm': inv_norm}, out=ch)

def contrast_adjustment(L, args):
    """Apply a mild S-curve contrast boost to the dynamic-range-stretched,
    black/white-normalised L channel (range [0,1] -- this runs after
    stretch_and_normalise_L, before the *100 scale to Lab lightness).

    The curve is a 3rd-order polynomial y = 0.5707x³ - 1.8298x² + 2.2592x
    (x in [0,1]) that lifts mid-tones while leaving the black and white
    points anchored: y(0)=0, y(1)=1 (to within the curve's own fit
    precision). The strength is controlled by args.contrast, which scales
    the deviation from the identity line. EDS images default to 1.6; EWS
    to 1.0.

    The curve was originally derived against stretched (asinh) data, not
    raw linear flux -- so it belongs here, immediately after the stretch,
    operating on the same [0,1] stretched domain it was designed for.

    Applied to L alone (not a*/b*, not B/G/R), so hue is not touched.

    The operation is done in-place via numexpr.
    """
    if args.contrast == 0:
        return

    if args.contrast is None:
        height, width = L.shape
        args.contrast = 1.6 if (height == 10200 and width == 10200) else 1.0

    c = args.contrast
    ne.evaluate(
        "c * (0.5707*L**3 - 1.8298*L**2 + 2.2592*L - L) + L",
        local_dict={'L': L, 'c': c}, out=L)

# ---------------------------------------------------------------------------
# CVD daltonization and cutout cropping
# ---------------------------------------------------------------------------

def _apply_cvd_daltonize(B, G, R, k, cvd_type):
    """Apply Fidaner+2005 daltonization on raw linear flux, before the
    dynamic-range stretch (see main()'s call order). The daltonization
    matrices are applied directly without any additional rescaling.
    B, G, R are modified in-place.

    Combined matrices D = I + E*(I-S)  (see full derivation in previous
    version of this function):
        D_d = [[ 1.50647, -0.50647, 0.00000],   deutan
               [ 0.00000,  1.00000, 0.00000],
               [-0.18125,  0.18125, 1.00000]]

        D_p = [[1.00000,  0.00000, 0.00000],     protan
               [0.51489,  0.48511, 0.00000],
               [0.61931, -0.61931, 1.00000]]

    k in [0,1] interpolates: D(k) = (1-k)*I + k*D_full.
    """
    k = float(np.clip(k, 0.0, 1.0))
    if k == 0.0:
        return

    if cvd_type == "deutan":
        a = np.float32(k * 0.50647)
        b = np.float32(k * 0.18125)
        R_new = ne.evaluate("(1.0 + a)*R - a*G",
                            local_dict={"R": R, "G": G, "a": a})
        B_new = ne.evaluate("B - b*R + b*G",
                            local_dict={"B": B, "R": R, "G": G, "b": b})
        R[:] = R_new
        B[:] = B_new
    else:  # protan
        a = np.float32(k * 0.51489)
        b = np.float32(k * 0.61931)
        G_new = ne.evaluate("a*R + (1.0 - a)*G",
                            local_dict={"R": R, "G": G, "a": a})
        B_new = ne.evaluate("B + b*R - b*G",
                            local_dict={"B": B, "R": R, "G": G, "b": b})
        G[:] = G_new
        B[:] = B_new

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
