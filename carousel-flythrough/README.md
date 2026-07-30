# Carousel Flythrough

A rendered mp4 animation of a camera flying through real 3D space along the
line of sight of the "Carousel" strong-lensing galaxy cluster (z≈0.49),
starting near redshift zero and flying out past z≈4.09 — the highest
confirmed redshift in the survey. Every galaxy and lensed-arc image on
screen is a real pixel cutout from the actual MUSE/HST imagery, placed at
its true sky position and true cosmological distance; nothing is a
synthetic glyph or symbol.

**Output:** `output/carousel_flythrough.mp4` (26s, 30fps, 1152×1152, ~6.9MB)

The camera flies straight down the lens's own axis: the scene origin is the
**main deflector**, catalog object `2444` — see "Camera axis" below.

Lensed-arc images render with their true detected shape (segmented from
`data/cutouts/*.fits` per-source cubelets) instead of a circular cutout — see
"Real arc shapes" below. The four Lyman-α emitters have no broadband light at
all and are built from the cutouts' continuum-subtracted line maps instead,
coloured by the observed wavelength of their own Lyman-α — see "Lyman-α
emitters" below. The cluster is a flat MUSE billboard only while it is far
away; as the camera closes in the billboard cross-dissolves into the
individual galaxies — see "Billboard cross-dissolve" below.

The frame is in standard astronomical orientation: **North up, East left**,
with `+x = West` and `+y = North` in scene coordinates.
`flythrough/check_registration.py` is a standing check of this and of the
billboard's alignment against the stamps; it writes
`output/registration_zoom.png`.

## Quick regenerate

```bash
cd flythrough
python3 prepare_data.py         # builds output/scene.npz (3D positions)
python3 prepare_imagery.py      # builds output/muse_rgb.png + output/stamps.npy (cutouts)
python3 check_registration.py   # asserts billboard/stamp alignment, writes the zoom PNG
python3 flythrough.py           # renders output/carousel_flythrough.mp4
```

Useful flags on `flythrough.py`:
- `--preview` — render a handful of still frames (`output/preview_frames/`) instead of the
  full video. Fast iteration loop for tuning camera/visual params. Combine with
  `--preview-frames N`, `--fig-size`, `--dpi`.
- `--no-labels` — suppress the per-source "Source N, z=..." fading text tags.
- `--no-hud` — suppress the corner `z / D_A / D_C` readout.
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
galaxies + 1 foreground dwarf + 1 foreground star) and `lensed_sources.py` (42 individual
lensed images; 40 of them, across 12 background sources with confirmed redshifts
z=0.96–4.09, are rendered — a 13th source, "Source 10", has no confirmed z and is
excluded).

For every object: comoving distance `D_C = cosmo.comoving_distance(z)` via
`FlatLambdaCDM(H0=69, Om0=0.3)` (matches the paper's assumed cosmology), and a transverse
(x, y) offset in Mpc = angular offset from the field center × the object's *own* `D_C`
(small-angle/tangent-plane approx, valid over this ~2′ field). Output: `output/scene.npz`.

Two conventions here are load-bearing:

- **`FIELD_CENTER_*` is the main deflector**, catalog object `2444` (`DEFLECTOR_ID`),
  looked up out of the redshift catalog. See "Camera axis" below for why, and for the two
  earlier choices that were wrong. Whatever this point is, the billboard has to be offset
  to match it — that coupling is the subject of `check_registration.py`.
- **`x = −ΔRA·cosδ·D_C`.** RA increases eastward and the display is East-left, so x runs
  *opposite* to RA. The original `x = +ΔRA·cosδ·D_C` mirrored every object position
  against the imagery it was cut from.

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
  shape, segmented from `data/cutouts/*.fits` — see "Real arc shapes" below — and the
  Lyman-α emitters get theirs from the same files' line maps, see "Lyman-α emitters".

MUSE stamps get a **per-object exposure lift** (`apply_exposure`) so faint high-z sources
aren't invisible next to bright cluster members. It is a *single scalar gain applied
equally to R, G and B* — `gain = clip(0.85 / p99_luminance, 1.0, 8.0)` — so hue and
saturation are preserved exactly, and anything already at full scale passes through
identical to the billboard pixels. Measured p99 luminance over the 78 objects inside the
MUSE footprint runs 0.09→1.00 with a median of 0.81, so most stamps are untouched and only
the faintest arcs are really boosted. This only affects the small stamps, never the full
billboard PNG.

That "identical to the billboard" property is not cosmetic — the billboard cross-dissolves
into these stamps, and the dissolve only reads as a dissolve if the pixels agree. The
previous version used a per-channel `(crop / p99) ** 0.42`, which renormalised even
already-bright galaxies up toward white and desaturated everything, since `(R/G)**0.42`
pulls every channel ratio toward 1.

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
  (`CLUSTER_D_C`, z=0.4895) as an `imshow` with computed `extent`, centred at
  `BILLBOARD_X/BILLBOARD_Y` — the MUSE frame's own centre, which is 10.9″ from the scene
  origin. Drawing it at the origin instead would slide the photo ~21% of its own
  half-width off the galaxies it contains. It cross-dissolves into the individual galaxies
  as the camera approaches (see below) rather than growing into a giant flat wall.
- **Every other object** (field galaxies, cluster members, foreground dwarf, each lensed
  image) is rendered as its own real stamp from step 2 via painter's algorithm (`imshow`,
  far-to-near depth order), scaled by real angular size × its own comoving distance so
  perspective growth is physically consistent.
- **`origin="lower"` on every `imshow`.** `muse-rgb.fits` has `PC2_2 = +5.556e-05`, i.e.
  pixel row +1 → Dec increases, so the array must be drawn bottom-up to put North up.
  matplotlib's default `origin="upper"` renders the whole field upside down.
- **Billboard cross-dissolve**: `billboard_alpha()` smoothsteps 1→0 as the camera's distance
  from the cluster plane closes from `DISSOLVE_START_DEPTH = 1400` to `DISSOLVE_END_DEPTH =
  600` Mpc (≈1.6 s → 4.3 s in), and every object whose light is already *in* that photo
  fades in on `1 − bb_alpha`. So the field reads as the flat MUSE picture from far away and
  as real 3D galaxies up close, with nothing drawn twice and nothing missing in between.
  `Object3D.dissolve` marks who participates, via `_in_billboard()`: an object counts only
  if its stamp came from the MUSE frame *and* it sits at or behind the cluster plane
  (`CLUSTER_D_C − FOREGROUND_MARGIN_MPC`). The three genuine foreground objects — FGD at
  367 Mpc, 2046 at 1104, 2292 at 1519 — and everything outside the MUSE footprint always
  draw at full alpha. That exemption matters: the camera passes FGD at `cam_z ≈ 367`, long
  before the dissolve starts, so gating it on the dissolve means it is never seen at all.
- **Near-object handling**: apparent size scales ~1/depth, so without limits, close objects
  would balloon into opaque full-frame walls. `MAX_EXT` caps displayed size; alpha dissolves
  out beyond that (`FADE_START_EXT`→`FADE_END_EXT`) so passing close reads as flying *through*
  something, not slamming into a card.
- **Lensed sources** get no synthetic highlight — just their real stamp and a text label.
  An earlier version drew a warm-gold glow circle behind each one; it read as bokeh on the
  distant arcs and as a flat tan disc on the near ones, and it is gone.
- **Labels**: `"Source N\nz=..."` fading text near each lensed source as the camera
  approaches (`_alpha_for_depth_fade`, tuned by `LABEL_FADE_IN_MPC`). Each source is
  legible for ~3.9 s (median, alpha > 0.15). There is a genuine label-free stretch
  around t ≈ 16.5 s: nothing in the catalog sits between z ≈ 1.7 and z = 3.086.
- **HUD**: corner readout of current `z`, `D_A` and `D_C` in Mpc. `D_A = D_C/(1+z)`,
  exact in a flat cosmology. It is *not* monotonic — it peaks at 1773 Mpc at
  t = 15.8 s (z = 1.605) and falls to 1394 Mpc by the end, so 39% of the flight
  counts down while the camera keeps moving forward. That is real, and `D_C` is
  printed under it partly to give the viewer a monotonic number to read it against.
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
- **CRVAL is not the centre of `muse-rgb.fits`.** `CRPIX ≈ (56.5, 94.5)` in a 340×348 frame,
  so CRVAL is 27.9″ off the middle of the picture. Treating it as the field centre (which
  this pipeline did originally) silently mis-registers the billboard against its own
  galaxies. Always go through `cutout_data.muse_frame_geometry()`.
- **FITS row order vs matplotlib.** `PC2_2 > 0` means row +1 → Dec increases, so these
  arrays need `origin="lower"`. Every `imshow` of MUSE-derived pixels must pass it, and the
  cached arrays in `stamps.npy` are kept in FITS row order — the spot-check PNGs in
  `output/stamps/` are `flipud`'d *only* on save so they look right in an image viewer.
- **Two independent mirror bugs cancel out under inspection.** The original code had East on
  the right in scene coordinates *and* North down in the imagery. Each is a mirror, so
  eyeballing a single frame can look merely "rotated" rather than wrong. Check them
  separately, with `check_registration.py` — when both conventions are right, every stamp
  sits on top of its own light in the billboard.
- **Moving the scene origin silently moves the billboard off its galaxies.** The billboard
  is one photo pinned at the cluster plane; every stamp is placed by RA/Dec offset from the
  origin. Change the origin without changing `BILLBOARD_X/Y` and the two drift apart with
  nothing in the renderer complaining. `check_registration.py` measures the bulk offset
  between the two placements and asserts it is under 0.02″.
- **Desaturating a colour makes it brighter.** Pulling a spectral hue off the gamut edge
  raises the channels that were low, so it raises luminance — the Lyman-α stamps' p99 goes
  0.591 → 0.804 on saturation alone. Any change to `LAE_SATURATION` has to be paired with
  the brightness normalisation, which is why that is derived from the arcs at build time
  rather than hard-coded.
- **"Too bright" is usually chroma, not luminance.** The Lyman-α stamps were measurably
  *dimmer* than the arcs and galaxies around them and still dominated the frame, because
  they were the only pure hues in a near-neutral picture. Measure the max/min channel ratio
  before reaching for a brightness knob.

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
using that shape as alpha — replacing the circular stamp for every lensed image
that has a cutout except `2a`, which has none and keeps the old circular path.
The Lyman-α emitters and `4e` are segmented the same way but out of the line
map rather than the broadband frame; they are handled in a second pass and
excluded from this one (`skip_labels`).

**One stamp per connected blob.** Where several catalogued images share a
connected footprint at that threshold — `{3a,3b,3c}`, `{5a,5b}`, `{8a,8b}`,
`{9a,9b}`, `{12a,12b}` — they are drawn as a single piece of sky, keyed by the
first of them, with the rest listed in `merged_labels` and dropped from the
stamp cache so the renderer can't draw the same pixels twice. So 40 lensed
images render as 34 stamps. That is a statement about footprints and not a
recount: the field still has 40 images, some of which happen to touch, the same
way an Einstein ring is still four images.

An earlier version split these shared blobs by nearest catalog centroid. That
carved one real blob into Voronoi wedges whose bounding boxes were wildly
elongated — it is where the old "`8a` is ~58:1 elongated" note came from. With
blobs kept whole the largest aspect ratio in the whole set is 3.07:1.

`ra`/`dec` on an arc stamp are the sky position of its **bounding-box centre**,
which is what the renderer centres the stamp on. That is not the same point as
any catalog centroid, so `flythrough.py::build_objects` takes arc positions
from the stamp rather than the catalog row; positioning an arc on its centroid
(as this originally did) offsets the pixels from where they belong.

Lensed-image RA/Dec now also comes from the cutout's own `CENTROIDS` table
(`sky_centroid.ra/dec`, fit directly from the data) rather than the
hand-transcribed `lensed_sources.py` table, for the 39 of 40 images that have one —
more accurate than the paper's rounded sexagesimal positions. `source8.fits`
contains an extra untranscribed image, `8d`, with no confirmed redshift;
excluded, same rule as Source 10.

Stamps are no longer forced square: `stamps.npy` entries carry
`half_width_arcsec`/`half_height_arcsec` separately (was a single
`radius_arcsec`), and `Object3D` in `flythrough.py` carries `half_w_mpc`/
`half_h_mpc` so elongated arcs render at their true aspect ratio instead of
being squashed back toward circular.

An earlier pass also resolved the render-bug half of the original critique (all
in `flythrough.py`): the billboard's rectangular edge is feathered; multiple
images of one source share a single "Source N" label instead of repeating it;
and the neutral placeholder used for the ~17 objects with no MUSE/HST coverage
is dim and small instead of a prominent blue-gray blob.

Not addressed (data-resolution limit, not a rendering bug): stamp resolution
still collapses to a handful of real-but-noisy pixels at the highest
redshifts (e.g. z≈3–4 images).

**Known open issue — field galaxies are still circles.** Arcs got real
segmented shapes, but catalog galaxies still use `_feathered_alpha`, a plain
feathered circle. That was invisible while the billboard covered the cluster;
now that the billboard dissolves into them, they read as soft bokeh discs in
the close approach (`cam_z ≈ 1700`). Two things would fix it, neither done:
give each stamp an alpha derived from its own light rather than a circle, and
borrow luminance from `hubble_f140w.fits` (0.07″/px vs MUSE's 0.2″/px) while
keeping MUSE colour, since a 2.5″ MUSE stamp is only ~25 px and is upscaled
~10× at closest approach.

## Camera axis (changed 2026-07-29)

`FIELD_CENTER_*` / `DEFLECTOR_ID` in `prepare_data.py`. The scene origin, and
therefore the axis the camera flies along, is **catalog object `2444`**
(z=0.48829, 90.985458, −35.968244): the cluster's main deflector, `L_a` in
`papers/carousel-model/sections/modeling.tex`, the galaxy its main power-law
lens profile is centred on. Flying at it means the flight ends looking straight
down the system's own axis of symmetry, with the central image `4e` 0.49″ off
axis.

Identified as `L_a` on three independent counts: 0.23″ from the system's
namesake coordinate `DESI-090.9854-35.9683`; the brightest cluster member in
integrated HST F140W flux (4022 vs 2202 for the runner-up, `2399`); 0.49″ from
image `4e`. Note `FGD` is *not* this object — that is the z=0.086 foreground
dwarf, 15″ away and unrelated to the lensing.

Two earlier choices, both wrong, for the record:

- `CRVAL1/CRVAL2`, which is not even the frame centre — `CRPIX` is ~(56.5,
  94.5) of a 340×348 grid, so CRVAL sits 27.9″ (−22.8″ RA, +16.0″ Dec) off the
  middle of the picture. It put the axis 6.7″ from Source 9 (every other source
  is 28–57″ out), which made Source 9 loom over the whole flight.
- the MUSE frame's geometric centre via `cutout_data.muse_frame_geometry()`.
  Defensible, and it is still what the billboard is placed on, but it is 10.9″
  from the deflector and centred on nothing in particular.

Moving the origin off the frame centre is what forced `BILLBOARD_X/BILLBOARD_Y`
and `check_registration.py` into existence — see the gotcha above.

## Lyman-α emitters (added 2026-07-29)

`LAE_SOURCES` and `build_linemap_stamps()` in `prepare_imagery.py`.

Sources 8, 11, 12 and 13 are the field's four LAEs, and Carousel Lens I says
three of them "are not apparent in imaging data and are only detected in MUSE
IFU data". Sampling their colour out of `muse-rgb.fits` — a broadband picture
in which they are by definition absent — therefore sampled whatever else lay
along that sightline. Two failure modes, both wrong in the same way: an image
buried in a cluster galaxy's light came out **gold** (`13e` on the BCG, `12e`,
`13d` in the halo of member `2521`), and an image on empty sky came out
**black** (`11b`, `11c`, `12d`, `13c`, all under RGB 0.01).

The cutouts hold the right data. Their `DATA` layer is the continuum-subtracted
narrow-band line map — for 12/13, the cube summed over 4962–4972 Å and smoothed
at σ = 1.5 px — i.e. the Lyman-α emission itself. So these stamps take shape and
luminance from the line flux (arcsinh-stretched), and colour from the observed
wavelength of their own Lyman-α:

| source | z | Lyα observed | colour |
|---|---|---|---|
| 12, 13 | 3.086 | 4967 Å | cyan-teal |
| 8 | 3.549 | 5530 Å | yellow-green |
| 11 | 4.090 | 6188 Å | orange |

so the colour of an emitter is a direct readout of its redshift. Set by
`LAE_COLOR_MODE`. The two alternatives, `"muse_hue"` (the old broadband hue) and
`"lya_blend"` (half way between), are still implemented but are **measured dead
ends** — see below.

**There is no true broadband colour to recover.** Measured straight off the MUSE
cube (`data/cube.fits/cube.fits` — a 3.5 GB IFU datacube nothing else in this
pipeline reads): three bands 4750–6100 / 6100–7600 / 7600–9300 Å, sky lines
rejected using the cube's own `STAT` plane, all three Lyα windows masked, 3–8 px
annulus background, uncertainty from ~150 blank apertures of matched area.

| source | b1 | b2 | b3 | total S/N |
|---|---|---|---|---|
| 4 (control — real galaxy, detected in imaging) | 0.955 | 1.309 | 1.300 | **13.2** |
| 8 | 0.095 | 0.100 | 0.062 | 0.9 |
| 11 | 0.025 | 0.032 | 0.006 | 0.1 |
| 12 | 0.037 | 0.069 | 0.010 | 0.6 |
| 13 | 0.156 | 0.455 | 0.483 | 4.3 |

Source 4 comes back at S/N 13, so the method works; none of the four LAEs does.
Source 13's 4.3 is entirely image `13e` (bands 1.09/2.91/3.31, ~30× every other
image of 13), which sits on the BCG — the same contamination that made `13e` gold
in the first place, and `4e` returns S/N 83 for the same reason. So `"muse_hue"`
does not sample the emitter; it samples whichever cluster galaxy happens to lie
along the sightline. Matches the survey paper, which has three of the four
invisible in imaging and detected in the IFU alone.

**Saturation and luminance floor.** The first version of these stamps read as
neon slabs, but not because they were bright: their 99th-percentile premultiplied
luminance was 0.591, *below* the real arcs (0.652) and the cluster galaxies
(0.835). It was chroma. The rest of the picture is near-neutral — `img_4c`, a
real arc, has mean RGB `[0.293, 0.293, 0.302]`, a max/min channel ratio of 1.0 —
while `img_12a` was `[0.058, 0.385, 0.275]`, a ratio of 6.6, with green pinned at
the top of the gamut. `LAE_SATURATION` 0.85 → 0.35 takes that ratio to 1.5. The
old `LAE_LUM_FLOOR = 0.30` was the other half: a hard pedestal under every pixel
above the S/N cut, so the blob had no profile, only an edge. 0.10 restores the
falloff at a cost of 8–14% of the rendered area (~4–7% linear), so the extension
won by moving to a line-map threshold survives.

The two have to move together, and not in the intuitive direction: desaturating
pushes every channel *toward white*, so it raises apparent luminance. At
saturation 0.35 the p99 goes 0.591 → 0.804 with no compensation — the fix for
"too bright" would have made them brighter. So `build_linemap_stamps` takes a
`target_p99` and rescales the finished stamps to match the median of the
broadband arcs built in the pass before it (×0.81 at current settings, printed at
build time). One scale for all of them, not per stamp: the brightness differences
*between* images of one source are magnification, which is the physics on
display. `CHROMA_INHERIT` entries (`4e`) are excluded from both the measurement
and the scale — see below.

The same switch fixes the "LAEs look too small" complaint, because it removes
the reason the threshold had to be high: a broadband cut has to stay at S/N 2
to avoid pulling in contaminating continuum, a line map does not.
`LAE_SN_THRESHOLD = 1.0` grows these footprints ~1.35–1.7× linearly (`13b`
54→150 px, `13c` 38→108 px, `12e` 17→38 px) with **no new blob merges** —
`{8a,8b}` and `{12a,12b}` stay, nothing else joins. Alpha is a soft ramp in S/N
(0.7→3.0) gated by the blob's own dilated region, rather than a blurred binary
mask, so the edge falls off the way the emission does.

`4e` rides along for a different reason. It is not an LAE; it is Source 4's
central image, sitting 0.49″ from the main deflector and drowned in its light,
so its broadband stamp was that galaxy's gold. It gets its shape from the line
map the same way, but its colour is inherited from its own source's clean
images `4a`–`4d` via `CHROMA_INHERIT` — including their mean *brightness*, not
just their hue, so that a demagnified central image does not come out brighter
than the outer images it is a counterpart of.

15 stamps take this path: 8→2 (after the `{8a,8b}` merge), 11→3, 12→4 (after
`{12a,12b}`), 13→5, plus `4e`.

**Image `7d` removed** in the same pass. The paper's source table gives its
redshift as "n.a. — Not firmly detected", because it "is both demagnified and
blended with the light of a cluster galaxy"; `source7.fits` has no detection for
it. It was therefore falling through to the 2.5″ circular MUSE fallback, whose
crop is centred 1.27″ from cluster member `2399` (z=0.487) — so the pipeline was
drawing a gold z≈0.49 elliptical at z=1.627, where it ballooned into the largest
object on screen mid-flight. Not recoverable from HST either: subtracting an
azimuthal-median model of `2399` from F140W leaves S/N = 0.7 at `7d`'s position,
inside the −0.8…+1.3 range of control apertures at the same radius. Source 7
goes 4 images → 3, total rendered images 41 → 40. `2a` is now the only lensed
image still on the circular path, and it is faint and small enough to leave.

## Likely next tweaks (if asked to refine)

- Camera speed/timing: `Z_START`, `Z_END`, `DURATION_S`, `ease_frac` in `flythrough.py`.
- FOV / how large things appear: `FOV_DEG`.
- When the billboard gives way to individual galaxies: `DISSOLVE_START_DEPTH`,
  `DISSOLVE_END_DEPTH`.
- Label timing/fade: `LABEL_FADE_IN_MPC` (currently 1400, ≈3.9 s per source) and the
  `fontsize` in the label `ax.text` (currently 11). Raising the fade further crowds the
  frame: 1400 puts at most 6 labels up at once, 2000 puts up 8.
- Stamp exposure/contrast for faint sources: `EXPOSURE_TARGET`, `EXPOSURE_MAX_GAIN` in
  `prepare_imagery.py`. Keep the gain floored at 1.0 or bright stamps stop matching the
  billboard and the cross-dissolve starts to show a seam.
- Stamp cutout size: `STAMP_RADIUS_ARCSEC` in `prepare_imagery.py` (currently 2.5″;
  re-run `prepare_imagery.py` after changing).
- Lyman-α emitter colour: `LAE_COLOR_MODE` (`"lya_wavelength"` / `"muse_hue"` /
  `"lya_blend"`, plus `LAE_BLEND_FRAC`) and `LAE_SATURATION` (currently 0.35) in
  `prepare_imagery.py`. `LAE_SATURATION` is safe to turn on its own — the brightness
  normalisation is derived at build time from the arcs, so it re-matches automatically.
  Note the other two colour modes are known-bad; see the continuum S/N table above.
- Lyman-α emitter brightness: `LAE_LUM_FLOOR` (currently 0.10) for the faint end, and
  the `target_p99` passed to `build_linemap_stamps` for the overall level.
- Lyman-α emitter size: `LAE_SN_THRESHOLD` (currently 1.0). Lowering it further keeps
  growing them, but re-check for new blob merges — the printed "contiguous footprint"
  lines from `prepare_imagery.py` are the audit trail.
- Output resolution/quality: `--fig-size`/`--dpi` flags, or the defaults in `main()`
  (`fig_size=7.2, dpi=160` for full render).

## Environment

Python 3.14, packages confirmed working: astropy 8.0.1, numpy 2.5.1, matplotlib 3.11.1,
imageio 2.37.4, Pillow 12.3.0, plus `imageio-ffmpeg` (pip-installed for its bundled ffmpeg
binary — not a system package).
