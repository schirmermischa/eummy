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

import sys, os, gc, cv2, threading
import numpy as np
import numexpr as ne
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.ndimage import distance_transform_edt
from concurrent.futures import ThreadPoolExecutor

from .eummy_io import (check_zeropoints, get_plate_scale, extract_wcs,
                        extract_tileID, find_images_in_directory, get_science_hdu,
                        read_zeropoint)


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
# Binned-mask helpers (shared by no-data detection and edge erosion)
# ---------------------------------------------------------------------------

def _row_stripes(n_rows, nthreads):
    """Split range(n_rows) into ~nthreads contiguous (start, stop) stripes."""
    n = max(1, min(nthreads, n_rows))
    edges = np.linspace(0, n_rows, n + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _bin_reduce(mask, BIN, reduce_all, nthreads):
    """Reduce a full-resolution boolean mask to BIN x BIN cells.

    reduce_all=True : cell is True iff ALL its pixels are True
    reduce_all=False: cell is True iff ANY of its pixels is True

    Cells at the right/bottom edge that extend beyond the image are reduced
    over their existing pixels only (equivalent to padding with the neutral
    element), so no padded full-size copy is ever made. Rows are processed
    in parallel stripes of whole binned rows; numpy reductions release the
    GIL, and each stripe writes a disjoint block of the output.
    """
    H, W = mask.shape
    Hb = (H + BIN - 1) // BIN
    Wb = (W + BIN - 1) // BIN
    Wm = (W // BIN) * BIN                 # width covered by whole cells
    op = np.logical_and.reduce if reduce_all else np.logical_or.reduce
    out = np.empty((Hb, Wb), dtype=bool)

    def work(b0, b1):
        r0, r1 = b0 * BIN, min(b1 * BIN, H)
        n_full = (r1 - r0) // BIN
        parts = []
        if n_full:                                       # view, no copy
            parts.append(op(mask[r0:r0 + n_full * BIN].reshape(n_full, BIN, W), axis=1))
        if r0 + n_full * BIN < r1:                       # partial last cell row
            parts.append(op(mask[r0 + n_full * BIN:r1], axis=0)[None, :])
        rows = parts[0] if len(parts) == 1 else np.vstack(parts)
        if Wm:
            out[b0:b1, :Wm // BIN] = op(rows[:, :Wm].reshape(b1 - b0, Wm // BIN, BIN), axis=2)
        if Wm < W:                                       # partial last cell column
            out[b0:b1, -1] = op(rows[:, Wm:], axis=1)

    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        for f in [ex.submit(work, b0, b1) for b0, b1 in _row_stripes(Hb, nthreads)]:
            f.result()
    return out


def _expand_combine(cells, BIN, full, combine, nthreads):
    """Expand a binned mask to full resolution (nearest neighbour) and
    combine it with the full-resolution mask `full` in the same pass:

    combine='and': result = expanded & full
    combine='or' : result = expanded | full

    Returns (result, n_changed) where n_changed counts pixels in which
    result differs from `full`. Expansion uses broadcast assignment of whole
    BIN-row blocks instead of a fancy-index gather (np.ix_), and expansion,
    combination and counting are fused per parallel row stripe, so each
    full-size array is traversed once.
    """
    H, W = full.shape
    Hb = cells.shape[0]
    result = np.empty((H, W), dtype=bool)
    counts = []
    lock = threading.Lock()

    def work(b0, b1):
        r0, r1 = b0 * BIN, min(b1 * BIN, H)
        # horizontal expansion on the small binned rows, cropped to W
        rows = np.repeat(cells[b0:b1], BIN, axis=1)[:, :W]
        n_full = (r1 - r0) // BIN
        if n_full:
            result[r0:r0 + n_full * BIN].reshape(n_full, BIN, W)[:] = rows[:n_full, None, :]
        if r0 + n_full * BIN < r1:
            result[r0 + n_full * BIN:r1] = rows[n_full]
        blk, ref = result[r0:r1], full[r0:r1]
        if combine == 'and':
            np.logical_and(blk, ref, out=blk)
        else:
            np.logical_or(blk, ref, out=blk)
        n = int(np.count_nonzero(blk)) - int(np.count_nonzero(ref))
        with lock:
            counts.append(abs(n))

    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        for f in [ex.submit(work, b0, b1) for b0, b1 in _row_stripes(Hb, nthreads)]:
            f.result()
    return result, sum(counts)

# ---------------------------------------------------------------------------
# Bad-pixel repair and channel blending
# ---------------------------------------------------------------------------

def _large_regions(bad, min_area, nthreads, BIN=8):
    """Return the pixels of `bad` that belong to a connected region of at
    least `min_area` pixels (8-connectivity).

    Used to tell zero-valued no-data areas (tile corners, strips -- large)
    apart from saturated star cores (also zero in all bands, but only
    ~5-20 px across), which must keep rendering white.

    Labelling runs on a BIN x BIN binned mask (a cell is set if ANY of its
    pixels is bad), avoiding a full-resolution int32 label array (~1.5 GB
    for a 19200^2 tile). Region area is estimated as n_cells * BIN^2, which
    overestimates the true area (partial cells count fully, and bad pixels
    within ~BIN of each other merge): a 20 px core can count as up to
    ~1000 px, so min_area should stay well above that. The returned mask is
    ANDed with `bad`, so only genuinely bad pixels are ever flagged.
    """
    from scipy.ndimage import label

    bad_b = _bin_reduce(bad, BIN, reduce_all=False, nthreads=nthreads)
    labels, n = label(bad_b, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return np.zeros_like(bad)
    big = np.bincount(labels.ravel()) * (BIN * BIN) >= min_area
    big[0] = False                                   # label 0 = background
    if not big.any():
        return np.zeros_like(bad)
    big_b = big[labels]
    labels = None
    out, _ = _expand_combine(big_b, BIN, bad, 'and', nthreads)
    return out

def repair_bad_pixels(B, G, R, L, args, thresh_scale=1.0):
    """Replace missing and anomalous pixel values before colour composition.

    Four repair passes are applied in sequence:

    1. Bad NISP pixels (zero OR NaN in any NIR band): replaced with the VIS
       luminance value L so the pixel contributes greyscale rather than a
       colour artefact.
    2. Bad VIS pixels (L zero OR NaN while all NIR bands are valid):
       replaced with the mean of the three NIR channels.
    3. Hot pixels (automatic, EWS only): pixels that are both absolutely
       bright and anomalously bright relative to their neighbours across
       channels are replaced with the local cross-channel average.  The VIS
       threshold is intentionally high to avoid clipping compact star cores.
    4. Any pixel still zero in any channel after the above is set to a large
       value (1e5) so that it renders as white rather than black in the
       final image.

    NaN is treated as equivalent to zero throughout steps 1-4: a pixel that
    is NaN in only one (or a few, but not all four) channels is repaired by
    borrowing from the valid channels exactly as a genuine zero pixel would
    be -- a single-band NaN (e.g. a cosmic-ray gap in one NIR band) should
    not blank out three otherwise-good channels. Only a pixel that is NaN
    in ALL FOUR channels (true no-data -- outside the survey footprint in
    every band) is rendered as black; see step 5. Additionally, pixels that
    are zero-or-NaN in all four channels (any mix) are rendered black if
    they belong to a connected region of at least args.nodata_min_area
    pixels -- no-data areas filled with zeros instead of NaN. Smaller such
    blobs (saturated star cores) still render white via step 4.

    thresh_scale rescales the two *absolute* flux thresholds (thresh1,
    thresh3) to match the actual data, combining two independent
    corrections computed in rescale_and_blend and multiplied together:
      - zeropoint: only when --no-zp-check, tracking the J-band zeropoint's
        deviation from the reference (30.1) as 10**((zp_measured-30.1)/2.5).
      - plate scale: always, tracking the I-band's plate scale deviation
        from the nominal 0.1"/pixel Euclid MER convention as
        (plate_scale/0.1)**2 (flux-per-pixel scales with pixel area).
    The two *ratio* thresholds (thresh2, thresh4) are NOT rescaled by
    either: they compare a channel to a scaled average of the other
    channels, which are already on the same flux scale, so that ratio is
    scale-invariant by construction -- rescaling it would be wrong, not
    just unnecessary.
    """
    print("Repairing bad pixels")

    mask = np.empty(L.shape, dtype=bool)

    # 0. Identify true no-data pixels: NaN in ALL FOUR channels (not just
    #    any one -- a pixel that's NaN in only some channels is repaired by
    #    borrowing from the valid ones in steps 1-2 below, same as a
    #    genuine zero pixel). Individual NaN values are replaced with plain
    #    0.0 so the existing zero-checks in steps 1-4 treat them exactly
    #    like a genuine bad/zero pixel -- no special-casing needed there.
    nan_mask = np.empty(L.shape, dtype=bool)
    ne.evaluate("(B != B) & (G != G) & (R != R) & (L != L)",
                local_dict={'B': B, 'G': G, 'R': R, 'L': L}, out=nan_mask)

    # 0b. No-data areas filled with zeros (or a mix of zeros and NaN) instead
    #     of pure NaN: a pixel is "bad" if each of the four bands is zero or
    #     NaN. Such pixels also occur in saturated star cores, which must keep
    #     rendering white (step 4), so only bad pixels in LARGE connected
    #     regions (>= --nodata-min-area) are treated as no-data. Pure all-NaN
    #     pixels stay no-data regardless of size (unchanged behaviour).
    #     Must run before the NaN -> 0 replacement below.
    if args.nodata_min_area > 0:
        ne.evaluate("((B != B) | (B == 0)) & ((G != G) | (G == 0)) & "
                    "((R != R) | (R == 0)) & ((L != L) | (L == 0))",
                    local_dict={'B': B, 'G': G, 'R': R, 'L': L}, out=mask)
        if mask.any():
            large = _large_regions(mask, args.nodata_min_area, args.nthreads)
            n_new = int((large & ~nan_mask).sum())
            if n_new:
                print(f"  {n_new:,} zero-valued no-data pixels (regions >= "
                      f"{args.nodata_min_area} px) will render black")
            nan_mask |= large
            large = None

    for ch in (B, G, R, L):
        ne.evaluate("where(ch != ch, 0.0, ch)", local_dict={'ch': ch}, out=ch)

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
        # thresh1/thresh3 (absolute) are scaled by thresh_scale to track the
        # data's actual photometric zeropoint; thresh2/thresh4 (ratios) are
        # not, since they're already scale-invariant -- see docstring.
        thresh1, thresh2, thresh3, thresh4 = 5 * thresh_scale, 5, 3 * thresh_scale, 20

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

    # 5. Zero out true no-data pixels (NaN in all four channels -- see step 0)
    #    so they render as black. Done after step 4 so any genuinely
    #    saturated/zero pixels are already white; only nan_mask pixels get
    #    overridden back to black here.
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
    # by more than 0.1% (relative, band-vs-band check). Separately, if the
    # I-band's own plate scale deviates from the nominal 0.1"/pixel Euclid
    # MER convention, --blackwhite/--pivot are corrected for that (absolute,
    # vs-nominal check) -- see below. Silent unless a correction is needed.
    #
    # get_plate_scale() returns None (with its own warning already printed)
    # if a band's header has no valid celestial WCS. If that's the I-band
    # itself, there is no reference to compare anything against, so plate-
    # scale checking is skipped entirely for all bands (assume no resampling
    # needed anywhere, and no --blackwhite/--pivot correction is possible).
    # If it's a non-reference band, that band alone is assumed to match the
    # reference (no resampling for it specifically).
    plate_scales = {band: get_plate_scale(path) for band, path in files_abs.items()}
    ps_i = plate_scales['I']
    zoom_factors = {'I': 1.0}
    plate_scale_factor = 1.0  # default: no correction possible/needed
    if ps_i is None:
        print("  WARNING: I-band has no valid celestial WCS — skipping plate-scale "
              "checking for all bands (assuming no resampling is needed anywhere).")
        zoom_factors.update({'Y': 1.0, 'J': 1.0, 'H': 1.0})
    else:
        for band in ('Y', 'J', 'H'):
            if plate_scales[band] is None:
                zoom_factors[band] = 1.0
                continue
            ratio = plate_scales[band] / ps_i
            if abs(ratio - 1.0) > 1e-3:
                zoom_factors[band] = ratio
                print(f"  {band}: {plate_scales[band]:.5f}\"/px vs I: {ps_i:.5f}\"/px "
                      f"→ resampling by factor {ratio:.6f}")
            else:
                zoom_factors[band] = 1.0

        # --blackwhite and --pivot's default values (and any values you pass
        # explicitly) are calibrated for flux-per-pixel at the nominal
        # Euclid MER plate scale (0.1"/pixel). Flux-per-pixel for a fixed
        # surface-brightness source scales with pixel AREA, i.e. with
        # plate_scale**2 -- e.g. at 0.3"/pixel (3x coarser), each pixel
        # covers 9x the sky area and so collects ~9x the flux for the same
        # underlying brightness. This is a purely geometric effect,
        # independent of and in addition to any photometric zero-point
        # correction (check_zeropoints), and the relative Y/J/H-vs-I
        # resampling above does NOT correct for it either -- that only
        # detects mismatches *between* the four bands, not an absolute
        # deviation from the nominal 0.1"/pixel convention shared by all
        # of them. So it's corrected here explicitly: --blackwhite (in
        # flux units) is scaled up by that same factor, and --pivot
        # (defined via 1/pivot in flux units) is scaled down by it, so
        # both stay pointed at the same physical brightness levels
        # regardless of the data's actual plate scale. This same
        # plate_scale_factor is also applied to repair_bad_pixels' absolute
        # hot-pixel thresholds further down -- the raw pixel DATA itself is
        # never corrected for this (unlike a zeropoint mismatch, nothing
        # here re-scales the actual flux values when all four bands share
        # the same non-nominal plate scale), so that correction must happen
        # unconditionally, not just when --no-zp-check is given.
        NOMINAL_PLATE_SCALE = 0.1
        plate_scale_factor = (ps_i / NOMINAL_PLATE_SCALE) ** 2
        if abs(plate_scale_factor - 1.0) > 1e-3:
            args.blackwhite = [v * plate_scale_factor for v in args.blackwhite]
            args.pivot = args.pivot / plate_scale_factor
            print(f"  I-band plate scale {ps_i:.5f}\"/px (nominal {NOMINAL_PLATE_SCALE}\"/px) "
                  f"→ scaling --blackwhite by {plate_scale_factor:.4f}× and "
                  f"--pivot by {1.0/plate_scale_factor:.4f}× "
                  f"(blackwhite={args.blackwhite}, pivot={args.pivot:.6f})")

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

    # repair_bad_pixels' absolute hot-pixel thresholds were calibrated for
    # data at the reference J-band zeropoint (30.1). If check_zeropoints ran
    # (the default), the flux-scaling step above already normalised every
    # band's data to that same reference-equivalent scale regardless of its
    # actual zeropoint (verified: with the correction applied, raw counts at
    # any zeropoint divide down to the same common value, to within
    # floating-point precision) -- so the thresholds are already correct as
    # given and must NOT be rescaled again here, or the same zeropoint
    # correction would be applied twice.
    # Only with --no-zp-check does the data remain on its raw, zeropoint-
    # dependent scale (--scales is used exactly as given, uncorrected), and
    # only then do the thresholds actually need to track the J-band
    # zeropoint's deviation from the reference to keep flagging the same
    # *physical* hot-pixel level.
    zp_thresh_scale = 1.0
    if args.no_zp_check:
        zp_j_measured = read_zeropoint(files_abs['J'])
        if zp_j_measured is not None:
            zp_thresh_scale = 10.0 ** ((zp_j_measured - 30.1) / 2.5)
            if abs(zp_thresh_scale - 1.0) > 0.01:
                print(f"  J-band zeropoint {zp_j_measured:.3f} (ref 30.1) — "
                      f"scaling hot-pixel thresholds by {zp_thresh_scale:.4f}× (zeropoint)")

    # Unlike the zeropoint case, plate_scale_factor applies unconditionally:
    # nothing in the pipeline corrects the raw pixel DATA for an absolute
    # plate-scale deviation from the 0.1"/pixel nominal (only --blackwhite/
    # --pivot get corrected, above) -- so the same factor must also be
    # applied here regardless of --no-zp-check.
    if abs(plate_scale_factor - 1.0) > 1e-3:
        print(f"  I-band plate scale {ps_i:.5f}\"/px — scaling hot-pixel "
              f"thresholds by {plate_scale_factor:.4f}× (plate scale)")

    thresh_scale = zp_thresh_scale * plate_scale_factor

    nan_mask = repair_bad_pixels(y_data, j_data, h_data, i_data, args,
                                  thresh_scale=thresh_scale)

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
# Edge erosion
# ---------------------------------------------------------------------------

def apply_edge_margin(nan_mask, margin, nthreads=1):
    """Grow the no-data mask by `margin` pixels into the valid data, so that
    pixels within `margin` of a no-data region also render black (ported from
    mosaic_eummy.py's --edge-margin).

    The "data edge" is the boundary of nan_mask (pixels NaN in all four
    bands, plus large zero-valued regions if --nodata-min-area is set; see
    repair_bad_pixels). The image frame itself is NOT treated as an edge --
    the distance transform only measures distance to no-data pixels inside
    the array -- so a tile without no-data regions is unchanged. This
    matches mosaic_eummy.py, where only uncovered regions inside the mosaic
    bounding box are eroded.

    As in mosaic_eummy.py, the distance transform runs on a binned mask for
    speed and memory (a full-resolution EDT of a 19200^2 tile would need a
    ~3 GB float64 array). A binned cell counts as no-data only if ALL its
    pixels are no-data, equivalent to mosaic_eummy rounding tile footprints
    outward. The bin factor is reduced for small margins (margin/BIN >= 20
    binned pixels; BIN=1, i.e. exact, for margin < 40).

    Binning error is one-sided: the masked zone is never wider than
    `margin`, but can be up to ~3*BIN pixels narrower (partially no-data
    cells count as data, plus cell-centre vs pixel distance). Measured on a
    synthetic slanted edge: 3 px short at margin=50 (BIN=2), 29 px short at
    margin>=300 (BIN=10). mosaic_eummy.py has the same behaviour.

    The full-resolution passes (binning, expansion + OR, counting) run in
    parallel row stripes over nthreads threads.

    Returns a new boolean mask (nan_mask OR edge zone); nan_mask is unchanged.
    """
    if margin <= 0:
        return nan_mask

    H, W = nan_mask.shape
    BIN = int(min(10, max(1, margin // 20)))

    # Binned no-data cells; also tells us cheaply whether there is any
    # no-data at all (a cell is set iff all its pixels are no-data).
    nodata_b = _bin_reduce(nan_mask, BIN, reduce_all=True, nthreads=nthreads)
    if not nodata_b.any():
        # Either no no-data at all, or only regions too small to fill a
        # single BIN x BIN cell -- the binned distance transform would not
        # see them anyway (same as mosaic_eummy). Also guards the
        # no-background case: distance_transform_edt then does not return
        # infinity but distances to a virtual point just outside the (0, 0)
        # corner, which would blank a quarter-disc of radius `margin` there.
        print(f"Edge margin {margin} px: no no-data region at {BIN}x binning — nothing to erode")
        return nan_mask
    if nodata_b.all():
        return nan_mask

    print(f"Edge margin {margin} px: distance transform at {BIN}x binning "
          f"({nodata_b.shape[1]}×{nodata_b.shape[0]})", flush=True)

    # Distance (in binned pixels) from each data cell to the nearest no-data
    # cell; no-data cells themselves have distance 0.
    edge_b = distance_transform_edt(~nodata_b) < (margin / BIN)
    nodata_b = None

    # Expand to full resolution, OR with the exact original no-data pixels
    # (binning only treats fully no-data cells as no-data, so partial cells
    # must not be lost), and count new pixels -- one fused parallel pass.
    edge_mask, n_new = _expand_combine(edge_b, BIN, nan_mask, 'or', nthreads)

    print(f"  {n_new:,} additional pixels masked "
          f"({100.0 * n_new / edge_mask.size:.2f}% of image)", flush=True)
    return edge_mask

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
