"""
Build the 3D scene catalog for the Carousel flythrough.

Reads:
  - papers/carousel-spectroscopic-survey/redshift_catalog_clean.txt
    (57 field/cluster galaxies + 1 foreground dwarf, with confident redshifts)
  - lensed_sources.py (43 lensed images of 13 background sources, transcribed
    from the paper's source table)

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
from cutout_data import load_lensed_image_centroids

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CATALOG_PATH = os.path.join(
    ROOT, "papers/carousel-spectroscopic-survey/redshift_catalog_clean.txt"
)
OUT_PATH = os.path.join(ROOT, "output/scene.npz")

# MUSE pointing center (from data/muse-rgb.fits CRVAL1/CRVAL2), also used
# as the field center for transverse offsets.
FIELD_CENTER_RA = 90.9957461
FIELD_CENTER_DEC = -35.97494928

CLUSTER_Z = 0.4895  # biweight central redshift from the paper

# Matches the paper's assumed cosmology (Sec. 3.3: Omega_m=0.3, h=0.69).
COSMO = FlatLambdaCDM(H0=69.0, Om0=0.3)


def sky_to_transverse_mpc(ra_deg, dec_deg, z):
    """Comoving transverse (x, y) offset in Mpc from the field center, at
    the object's own comoving distance (proper transverse comoving distance,
    i.e. angular offset * D_C, using a small-angle/tangent-plane approximation
    valid over this ~2' field)."""
    center = SkyCoord(FIELD_CENTER_RA * u.deg, FIELD_CENTER_DEC * u.deg)
    obj = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    dra = (obj.ra - center.ra).wrap_at(180 * u.deg).radian * np.cos(center.dec.radian)
    ddec = (obj.dec - center.dec).radian
    d_c = COSMO.comoving_distance(z).to(u.Mpc).value
    x = dra * d_c
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
        cluster_z=CLUSTER_Z,
        cluster_d_c=COSMO.comoving_distance(CLUSTER_Z).to(u.Mpc).value,
        **{f"field_{k}": v for k, v in field_arrs.items()},
        **{f"lensed_{k}": v for k, v in lensed_arrs.items()},
    )
    return field_rows, lensed_rows


def summarize(field_rows, lensed_rows):
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
