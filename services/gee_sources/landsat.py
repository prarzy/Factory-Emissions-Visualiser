import ee
from datetime import datetime, timedelta

from services.utils.gee_auth import ensure_ee_initialized


def fetch_lst(lat: float, lon: float):
    ensure_ee_initialized()

    point = ee.Geometry.Point(lon, lat)
    region = point.buffer(10_000).bounds()
    today = datetime.utcnow().date()
    start = today - timedelta(days=90)

    coll = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterDate(str(start), str(today))
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
        .select("ST_B10")
        .map(lambda img: (
            img.multiply(0.00341802)
            .add(149)
            .subtract(273.15)
            .copyProperties(img, img.propertyNames())
        ))
    )

    if coll.size().getInfo() == 0:
        raise RuntimeError("No cloud-free scenes in the last 90 days.")

    lst_celsius = coll.median()

    url = lst_celsius.getDownloadURL({
        "scale": 100,
        "region": region,
        "format": "NPY"
    })
    return url
