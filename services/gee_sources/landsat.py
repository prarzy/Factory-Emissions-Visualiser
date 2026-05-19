import ee
from datetime import datetime, timedelta, timezone

from services.utils.gee_auth import ensure_ee_initialized
from services.gee_sources.indices import compute_ndvi, compute_ndbi, create_industrial_mask


def fetch_lst(lat: float, lon: float):
    ensure_ee_initialized()

    point = ee.Geometry.Point(lon, lat)
    region = point.buffer(10_000).bounds()
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=90)

    # Select thermal band plus the surface reflectance bands needed for NDVI/NDBI
    coll = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterDate(str(start), str(today))
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
        .select(["ST_B10", "SR_B4", "SR_B5", "SR_B6"])
        .map(lambda img: (
            # Add a new LST_Celsius band from ST_B10 without touching SR bands
            img.addBands(
                img.select("ST_B10")
                .multiply(0.00341802)
                .add(149)
                .subtract(273.15)
                .rename("LST_Celsius")
            )
            .copyProperties(img, img.propertyNames())
        ))
    )

    if coll.size().getInfo() == 0:
        raise RuntimeError("No cloud-free scenes in the last 90 days.")

    # Median composite across all bands (LST + SR bands)
    composite = coll.median()

    # Compute spectral indices and build the industrial area mask
    #   NDVI < 0.3  → low vegetation
    #   NDBI > 0.2  → high built-up reflectance
    composite = compute_ndbi(compute_ndvi(composite))
    mask = create_industrial_mask(composite)

    # Apply mask to LST — only industrial pixels survive; rest become NaN
    lst_masked = composite.select("LST_Celsius").updateMask(mask)

    url = lst_masked.getDownloadURL({
        "scale": 100,
        "region": region,
        "format": "NPY",
    })
    return url
