"""
Build the 3D scene catalog for the Carousel flythrough.

Reads:
  - papers/carousel-spectroscopic-survey/redshift_catalog_clean.txt
    (57 field/cluster galaxies + 1 foreground dwarf, with confident redshifts)
  - lensed_sources.py (42 lensed images of 13 background sources, transcribed
    from the paper's source table; 40 of them, of 12 sources, are rendered -
    Source 10 has no confirmed redshift, see the note there)

Produces a flat array of objects, each with a 3D comoving position (Mpc) and a
category tag, saved to output/scene.npz for the renderer to consume.

Coordinate convention: camera space, +z along the line of sight (increasing
redshift/distance), x/y transverse comoving offset from the field center.
"""
import os

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from lensed_sources import LENSED_IMAGES, source_label
from cutout_data import load_lensed_image_centroids, muse_frame_geometry

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CATALOG_PATH = os.path.join(
    ROOT, "papers/carousel-spectroscopic-survey/redshift_catalog_clean.txt"
)
OUT_PATH = os.path.join(ROOT, "output/scene.npz")

# The axis the camera flies along: the main deflector of the lens, catalog
# object 2444. This is L_a in papers/carousel-model, the galaxy its main
# power-law lens profile is centred on (sections/modeling.tex). It is the
# obvious thing to fly at - it is what does the lensing, and the central
# image 4e sits 0.49" from it, so the flight ends looking straight down the
# system's own axis of symmetry.
#
# Identified as L_a on three counts: 0.23" from the system's namesake
# coordinate DESI-090.9854-35.9683; brightest cluster member in integrated
# HST F140W flux (4022 vs 2202 for the runner-up); 0.49" from image 4e.
# Note FGD is *not* this object - that is the z=0.086 foreground dwarf from
# the survey paper, 15" away and unrelated to the lensing.
#
# Two earlier choices, both wrong, for the record: CRVAL1/CRVAL2 (which is
# not the frame centre at all - CRPIX is ~(56.5, 94.5) of a 340x348 grid, so
# CRVAL sits ~28" off the middle, and it put the axis almost exactly through
# Source 9), then the MUSE frame's geometric centre (defensible, but 10.9"
# from the deflector and centred on nothing in particular).
DEFLECTOR_ID = "2444"

# Frame geometry is still needed for the billboard's true footprint, and its
# centre is still needed to *place* that billboard, which is no longer at the
# scene origin. See BILLBOARD_X/Y in flythrough.py.
MUSE_CENTER_RA, MUSE_CENTER_DEC, MUSE_HALF_W_ARCSEC, MUSE_HALF_H_ARCSEC = (
    muse_frame_geometry()
)


def _catalog_coord(object_id):
    """(ra_deg, dec_deg) for one object in the redshift catalog.

    Deliberately does not go through load_field_catalog(), which calls
    sky_to_transverse_mpc(), which reads FIELD_CENTER_* - the very thing this
    is used to define.
    """
    with open(CATALOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            oid, _z, ra_s, dec_s, _qop, _inst = line.split(",")
            if oid == object_id:
                c = SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg))
                return c.ra.deg, c.dec.deg
    raise KeyError(f"{object_id} not found in {CATALOG_PATH}")


FIELD_CENTER_RA, FIELD_CENTER_DEC = _catalog_coord(DEFLECTOR_ID)

CLUSTER_Z = 0.4895  # biweight central redshift from the paper

# Matches the paper's assumed cosmology (Sec. 3.3: Omega_m=0.3, h=0.69).
COSMO = FlatLambdaCDM(H0=69.0, Om0=0.3)


def sky_to_transverse_mpc(ra_deg, dec_deg, z):
    """Comoving transverse (x, y) offset in Mpc from the field center (the
    main deflector, see DEFLECTOR_ID), at the object's own comoving distance (proper transverse comoving distance,
    i.e. angular offset * D_C, using a small-angle/tangent-plane approximation
    valid over this ~2' field).

    Orientation is standard astronomical: +x is West and +y is North, so the
    rendered frame reads North-up / East-left, the same way the MUSE pixels
    do. Note the sign on x: RA increases *eastward*, so East-left means x
    runs opposite to RA. Getting this wrong mirrors every object position
    against the imagery it was cut from, which is what the first version of
    this pipeline did.
    """
    center = SkyCoord(FIELD_CENTER_RA * u.deg, FIELD_CENTER_DEC * u.deg)
    obj = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    dra = (obj.ra - center.ra).wrap_at(180 * u.deg).radian * np.cos(center.dec.radian)
    ddec = (obj.dec - center.dec).radian
    d_c = COSMO.comoving_distance(z).to(u.Mpc).value
    x = -dra * d_c
    y = ddec * d_c
    return x, y, d_c


def load_field_catalog():
    rows = []
    with open(CATALOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            oid, z, ra_s, dec_s, qop, inst = line.split(",")
            z = float(z)
            c = SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg))
            x, y, d_c = sky_to_transverse_mpc(c.ra.deg, c.dec.deg, z)

            if z < 0.01:
                category = "star"  # object 2357, z~0.00006, a foreground star
            elif oid == "FGD":
                category = "foreground_dwarf"
            elif abs(z - CLUSTER_Z) < 3000 / 299792.458 * (1 + CLUSTER_Z):
                # within ~3000 km/s of cluster redshift -> cluster member
                category = "cluster_member"
            else:
                category = "field"

            rows.append(
                dict(
                    label=oid,
                    ra=c.ra.deg,
                    dec=c.dec.deg,
                    z=z,
                    x=x,
                    y=y,
                    d_c=d_c,
                    category=category,
                    instrument=inst,
                )
            )
    return rows


def load_lensed_sources():
    # data/cutouts/*.fits carries a WCS-fit centroid (CENTROIDS table) for
    # most images, fit directly from the data rather than transcribed by
    # hand from the paper's sexagesimal table - more accurate where present.
    # Only used for position; redshift still comes from LENSED_IMAGES, since
    # the cutouts carry no redshift info. Note: source8.fits also contains
    # an untranscribed "8d" image with no confirmed redshift - naturally
    # excluded here since we iterate LENSED_IMAGES, not the cutout catalog.
    cutout_centroids = load_lensed_image_centroids()

    rows = []
    for image_label, ra_s, dec_s, z in LENSED_IMAGES:
        if z is None:
            continue  # Source 10: no confirmed redshift, excluded
        cutout = cutout_centroids.get(image_label)
        if cutout is not None:
            ra_deg, dec_deg = cutout["ra"], cutout["dec"]
        else:
            c = SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg))
            ra_deg, dec_deg = c.ra.deg, c.dec.deg
        x, y, d_c = sky_to_transverse_mpc(ra_deg, dec_deg, z)
        rows.append(
            dict(
                label=image_label,
                source=source_label(image_label),
                ra=ra_deg,
                dec=dec_deg,
                z=z,
                x=x,
                y=y,
                d_c=d_c,
                category="lensed_image",
            )
        )
    return rows


def build_scene():
    field_rows = load_field_catalog()
    lensed_rows = load_lensed_sources()

    def stack(rows, keys):
        return {k: np.array([r[k] for r in rows]) for k in keys}

    field_arrs = stack(
        field_rows, ["label", "ra", "dec", "z", "x", "y", "d_c", "category", "instrument"]
    )
    lensed_arrs = stack(
        lensed_rows, ["label", "source", "ra", "dec", "z", "x", "y", "d_c", "category"]
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(
        OUT_PATH,
        field_center_ra=FIELD_CENTER_RA,
        field_center_dec=FIELD_CENTER_DEC,
        deflector_id=DEFLECTOR_ID,
        muse_center_ra=MUSE_CENTER_RA,
        muse_center_dec=MUSE_CENTER_DEC,
        cluster_z=CLUSTER_Z,
        cluster_d_c=COSMO.comoving_distance(CLUSTER_Z).to(u.Mpc).value,
        **{f"field_{k}": v for k, v in field_arrs.items()},
        **{f"lensed_{k}": v for k, v in lensed_arrs.items()},
    )
    return field_rows, lensed_rows


def summarize(field_rows, lensed_rows):
    print(
        f"Camera axis: object {DEFLECTOR_ID} at "
        f"{FIELD_CENTER_RA:.6f}, {FIELD_CENTER_DEC:.6f} (the main deflector)"
    )
    dra = (MUSE_CENTER_RA - FIELD_CENTER_RA) * 3600 * np.cos(np.radians(FIELD_CENTER_DEC))
    ddec = (MUSE_CENTER_DEC - FIELD_CENTER_DEC) * 3600
    print(f"  MUSE frame centre is {dra:+.2f}\" RA, {ddec:+.2f}\" Dec from it\n")

    cats = {}
    for r in field_rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print("Field catalog:", len(field_rows), "objects")
    for k, v in cats.items():
        print(f"  {k}: {v}")
    zs = [r["z"] for r in field_rows]
    print(f"  z range: {min(zs):.5f} - {max(zs):.5f}")

    sources = sorted(set(r["source"] for r in lensed_rows), key=lambda s: int(s))
    print(f"\nLensed images: {len(lensed_rows)} images of {len(sources)} sources")
    for s in sources:
        zs = [r["z"] for r in lensed_rows if r["source"] == s]
        n_images = len(zs)
        print(f"  Source {s}: z={zs[0]:.3f}, {n_images} images")


if __name__ == "__main__":
    field_rows, lensed_rows = build_scene()
    summarize(field_rows, lensed_rows)
    print(f"\nSaved scene to {OUT_PATH}")
