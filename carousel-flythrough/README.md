# Carousel Flythrough

A rendered mp4 animation of a camera flying through real 3D space along the
line of sight of the "Carousel" strong-lensing galaxy cluster (z≈0.49),
starting near redshift zero and flying out past z≈4.09 — the highest
confirmed redshift in the survey. Every galaxy and lensed-arc image on
screen is a real pixel cutout from the actual MUSE/HST imagery, placed at
its true sky position and true cosmological distance; nothing is a
synthetic glyph or symbol.

**Output:** `output/carousel_flythrough.mp4` (26s, 30fps, 1152×1152, ~7.5MB)

Lensed-arc images now render with their true detected shape (segmented from
`data/cutouts/*.fits` per-source cubelets) instead of a circular cutout — see
"Real arc shapes" below.

## Quick regenerate

```bash
cd flythrough
python3 prepare_data.py      # builds output/scene.npz (3D positions)
python3 prepare_imagery.py   # builds output/muse_rgb.png + output/stamps.npy (cutouts)
python3 flythrough.py        # renders output/carousel_flythrough.mp4
```

Useful flags on `flythrough.py`:
- `--preview` — render a handful of still frames (`output/preview_frames/`) instead of the
  full video. Fast iteration loop for tuning camera/visual params. Combine with
  `--preview-frames N`, `--fig-size`, `--dpi`.
- `--no-labels` — suppress the per-source "Source N, z=..." fading text tags.
- `--no-hud` — suppress the corner `z = ... / D_C = ... Mpc` readout.
- `--out PATH` — write the mp4 somewhere other than the default.

Labels/HUD can also be turned off by default by flipping `SHOW_LABELS` /
`SHOW_HUD` at the top of `flythrough.py`.

`prepare_data.py` and `prepare_imagery.py` only need to be re-run if you change the
underlying catalogs, cosmology, or stamp/cutout logic — `flythrough.py` alone is
enough to re-render if you're only tuning camera/visual parameters, since it just
reads their cached outputs (`scene.npz`, `muse_rgb.png`, `stamps.npy`).

## Pipeline (3 scripts, in `flythrough/`)

### 1. `prepare_data.py` — 3D scene catalog
Reads `papers/carousel-spectroscopic-survey/redshift_catalog_clean.txt` (57 field/cluster
galaxies + 1 foreground dwarf + 1 foreground star) and `lensed_sources.py` (43 individual
lensed images across 12 background sources with confirmed redshifts z=0.96–4.09; a 13th
source, "Source 10", has no confirmed z and is excluded).

For every object: comoving distance `D_C = cosmo.comoving_distance(z)` via
`FlatLambdaCDM(H0=69, Om0=0.3)` (matches the paper's assumed cosmology), and a transverse
(x, y) offset in Mpc = angular offset from the field center × the object's *own* `D_C`
(small-angle/tangent-plane approx, valid over this ~2′ field). Output: `output/scene.npz`.

Categorization logic worth knowing about if you touch this file:
- `z < 0.01` → tagged `"star"` (object 2357 is a foreground Milky Way star, not a galaxy —
  excluded from rendering in `flythrough.py`).
- within ~3000 km/s of `CLUSTER_Z = 0.4895` → `"cluster_member"`.
- everything else → `"field"`.

`lensed_sources.py` holds `LENSED_IMAGES`, hand-transcribed from the "Full Source List"
table in `papers/carousel-spectroscopic-survey/main.tex`. If the paper's table changes,
this needs to be re-transcribed by hand — there's no automated LaTeX table parser.

### 2. `prepare_imagery.py` — real cutout stamps
Builds two things:
- **`output/muse_rgb.png`** — the full `data/muse-rgb.fits` frame, saved byte-for-byte as
  given (it's already a display-ready linear-scale picture per the companion
  `muse-rgb.pdf`; deliberately *not* re-stretched).
- **`output/stamps.npy`** — a dict of small RGBA cutouts, one per catalog galaxy and one per
  individual lensed image, each centered on that object's true RA/Dec. Preference order:
  MUSE RGB (true color, 0.2″/px) → HST F140W (grayscale fallback, 0.07″/px, arcsinh-stretched)
  → a small dim neutral placeholder for the ~17 objects covered by neither footprint. Field
  galaxies get a feathered circular alpha mask. Lensed images instead get their true detected
  shape, segmented from `data/cutouts/*.fits` — see "Real arc shapes" below.

MUSE stamps get a **per-object** exposure boost (`crop / percentile99 clipped, then **0.42
gamma`) so faint high-z sources aren't invisible next to bright cluster members — this only
affects the small stamps, never the full billboard PNG.

A handful of sample stamps are dumped to `output/stamps/*.png` for visual spot-checking.

### 3. `flythrough.py` — the renderer
Custom perspective-projection camera built directly on matplotlib (not mplot3d — needed
billboarded images + depth-based alpha fades + HUD text together). Key mechanics:

- **Camera path**: dollies along +z (comoving distance) from `Z_START=40` to `Z_END=7550` Mpc
  with smoothstep ease-in/out at both ends, linear cruise between. `camera_z_of_frame()`
  drives this; `build_redshift_lookup()` inverts distance→z via a precomputed interpolation
  grid (avoids per-frame astropy cosmology calls).
- **Projection**: standard pinhole model, `project()` / inlined in `render_frame()`, narrow
  `FOV_DEG = 8/60` (8 arcmin) telephoto-like FOV sized to the data's actual angular scale —
  a normal-photo FOV would shrink everything to sub-pixel specks.
- **Cluster billboard**: the full MUSE image placed at its true distance
  (`CLUSTER_D_C`, z=0.4895) as an `imshow` with computed `extent`; capped max size + fade as
  the camera gets very close, rather than growing into a giant flat wall.
- **Every other object** (field galaxies, cluster members, foreground dwarf, each lensed
  image) is rendered as its own real stamp from step 2 via painter's algorithm (`imshow`,
  far-to-near depth order), scaled by real angular size × its own comoving distance so
  perspective growth is physically consistent.
- **Near-object handling**: apparent size scales ~1/depth, so without limits, close objects
  would balloon into opaque full-frame walls. `MAX_EXT` caps displayed size; alpha dissolves
  out beyond that (`FADE_START_EXT`→`FADE_END_EXT`) so passing close reads as flying *through*
  something, not slamming into a card.
- **Lensed sources** get one extra cue on top of their real stamp: a soft warm-gold glow
  circle behind it (`(1.0, 0.82, 0.35)`, depth-modulated alpha) so the scientific highlights
  are easy to spot — the underlying stamp shape/color is never altered.
- **Labels**: `"Source N\nz=..."` fading text near each lensed source as the camera
  approaches/passes (`_alpha_for_depth_fade`, tuned by `LABEL_FADE_IN_MPC`/`LABEL_FADE_OUT_MPC`).
- **HUD**: corner readout of current `z` and `D_C` in Mpc.
- **Starfield**: a small fixed set of background points at large fixed distance
  (`build_starfield`), purely atmospheric — unrelated to any catalog data.
- **Encoding**: raw RGBA canvas buffer per frame (`fig.canvas.buffer_rgba()`) → `imageio` with
  `imageio-ffmpeg`'s bundled binary, `libx264` codec.

## Gotchas / things that bit us during development

- **matplotlib facecolor**: `fig.savefig(path, facecolor="black")` only overrides the color
  for that one save call — it does *not* persist on the `Figure`. Since the mp4 path reads
  the raw canvas buffer (`fig.canvas.buffer_rgba()`) instead of calling `savefig`, the black
  background has to be set at `plt.subplots(..., facecolor="black")` creation time, or frames
  come out with a default (often white/transparent-composited-wrong) background.
- **No system ffmpeg was present** — `imageio-ffmpeg` was pip-installed specifically because it
  bundles its own working ffmpeg binary. If this environment changes, `pip install
  imageio-ffmpeg` again before rendering.
- **`redshift_catalog_clean.txt`** is a pre-cleaned CSV (id, z, RA sexagesimal, Dec
  sexagesimal, QOP, instrument) already sitting in the papers directory — not something this
  project generated.
- **Object 2357** (z≈0.00006, `"star"` category) is deliberately excluded from rendering in
  `flythrough.py` (`build_objects`) as a foreground Milky Way star, not a galaxy.
- **Source 10** has two lensed images but no confirmed spectroscopic redshift in the paper —
  excluded entirely (`load_lensed_sources` skips `z is None` rows).

## Real arc shapes (added 2026-07-29)

The first render stamped every object — including lensed arcs — into a small
feathered *circle*, flattening arc morphology into bokeh dots (see git history
for the original critique). The user then supplied `data/cutouts/*.fits`: one
FITS cubelet per source (or small source group), with `DATA`/`STAT`/`MASK`
(hot pixels + unrelated contaminating sources, not the source's own shape)
/`CENTROIDS` (per-image WCS-fit position)/`PSF` extensions, continuum-subtracted
and pixel-aligned with `data/muse-rgb.fits` (confirmed: same CD matrix and
CRVAL, integer pixel offset only — see `cutout_data.py`).

`prepare_imagery.py::build_lensed_image_masks()` now segments each lensed
image's true footprint from its own `S/N = DATA/sqrt(STAT)` map (threshold
`ARC_SN_THRESHOLD = 2.0`, `MASK==0` pixels excluded, connected-component
labeled), lightly feathers the binary mask, and crops real MUSE RGB color
using that shape as alpha — replacing the circular stamp for the 39 of 41
lensed images that have a cutout (`2a` and `7d` have no cutout and keep the
old circular path). Where multiple images in one file share a connected blob
at that threshold (`{3a,3b,3c}`, `{5a,5b}`, `{9a,9b}`, `{12a,12b}` — raising
the threshold does not cleanly separate these without shrinking real extent),
pixels are split by nearest catalog centroid instead.

Lensed-image RA/Dec now also comes from the cutout's own `CENTROIDS` table
(`sky_centroid.ra/dec`, fit directly from the data) rather than the
hand-transcribed `lensed_sources.py` table, for the 40 images that have one —
more accurate than the paper's rounded sexagesimal positions. `source8.fits`
contains an extra untranscribed image, `8d`, with no confirmed redshift;
excluded, same rule as Source 10.

Stamps are no longer forced square: `stamps.npy` entries carry
`half_width_arcsec`/`half_height_arcsec` separately (was a single
`radius_arcsec`), and `Object3D` in `flythrough.py` carries `half_w_mpc`/
`half_h_mpc` so elongated arcs (e.g. image 8a is ~58:1 elongated) render at
their true aspect ratio instead of being squashed back toward circular.

Same pass also resolved the render-bug half of the original critique (all in
`flythrough.py`): cluster members are no longer drawn both as the billboard
*and* individually at the same depth (members now only render once the
billboard has faded); the billboard's rectangular edge is feathered; the
lensed-source glow is a subtle accent, not a dominant flat disc; multiple
images of one source share a single "Source N" label instead of repeating it;
and the neutral placeholder used for the ~17 objects with no MUSE/HST coverage
is now dim and small instead of a prominent blue-gray blob.

Not addressed (data-resolution limit, not a rendering bug): stamp resolution
still collapses to a handful of real-but-noisy pixels at the highest
redshifts (e.g. z≈3–4 images).

## Likely next tweaks (if asked to refine)

- Camera speed/timing: `Z_START`, `Z_END`, `DURATION_S`, `ease_frac` in `flythrough.py`.
- FOV / how large things appear: `FOV_DEG`.
- Glow color/strength on lensed sources: the `glow_ext`/`glow_alpha`/color block in
  `render_frame()`.
- Label timing/fade: `LABEL_FADE_IN_MPC`, `LABEL_FADE_OUT_MPC`.
- Stamp exposure/contrast for faint sources: the `0.42` gamma and 99th-percentile norm in
  `prepare_imagery.py::extract_stamp`.
- Stamp cutout size: `STAMP_RADIUS_ARCSEC` in `prepare_imagery.py` (currently 2.5″;
  re-run `prepare_imagery.py` after changing).
- Output resolution/quality: `--fig-size`/`--dpi` flags, or the defaults in `main()`
  (`fig_size=7.2, dpi=160` for full render).

## Environment

Python 3.14, packages confirmed working: astropy 8.0.1, numpy 2.5.1, matplotlib 3.11.1,
imageio 2.37.4, Pillow 12.3.0, plus `imageio-ffmpeg` (pip-installed for its bundled ffmpeg
binary — not a system package).
