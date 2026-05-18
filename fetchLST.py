import ee
import numpy as np
from datetime import datetime, timedelta


def fetch_lst(lat: float, lon: float):
    
    credentials = ee.ServiceAccountCredentials(
        "imsukudu24@gmail.com", "key.json"
    )
    ee.Initialize(credentials, project="careful-drummer-462304-u9")

    point   = ee.Geometry.Point(lon, lat)
    region  = point.buffer(10_000).bounds()          # radius 10 km → 20 km square
    today   = datetime.utcnow().date()
    start   = today - timedelta(days=90)

    coll = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterDate(str(start), str(today))
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
        .select("ST_B10")                            # LST band in Kelvin × 0.00341802 + 149
        .map(lambda img: (
            img.multiply(0.00341802)
            .add(149)                             # DN → Kelvin
            .subtract(273.15)                     # Kelvin → °C
            .copyProperties(img, img.propertyNames())  # keep metadata
        ))
    )

    if coll.size().getInfo() == 0:
        raise RuntimeError("No cloud‑free scenes in the last 30 days.")

    
    lst_celsius = coll.median()

    
    url = lst_celsius.getDownloadURL({
        "scale": 100,                # ≈ 100 m/pixel → manageable array
        "region": region,
        "format": "NPY"
    })
    return url
