import ee


def compute_ndvi(image: ee.Image) -> ee.Image:
    """Compute NDVI using Landsat 9 SR_B5 (NIR) and SR_B4 (Red).

    NDVI = (NIR - Red) / (NIR + Red)
    Uses ee.Image.normalizedDifference which handles zero-division.
    Adds the result as a new 'NDVI' band.
    """
    ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    return image.addBands(ndvi)


def compute_ndbi(image: ee.Image) -> ee.Image:
    """Compute NDBI using Landsat 9 SR_B6 (SWIR1) and SR_B5 (NIR).

    NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    Uses ee.Image.normalizedDifference which handles zero-division.
    Adds the result as a new 'NDBI' band.
    """
    ndbi = image.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
    return image.addBands(ndbi)


def create_industrial_mask(image: ee.Image) -> ee.Image:
    """Create a boolean mask for built-up / industrial areas.

    Industrial areas are characterised by:
      - low vegetation: NDVI < 0.3
      - built-up reflectance: NDBI > 0.0

    The caller must have already added NDVI and NDBI bands (e.g. via
    compute_ndvi and compute_ndbi) before passing the image here.
    """
    ndvi = image.select('NDVI')
    ndbi = image.select('NDBI')
    return ndvi.lt(0.3).And(ndbi.gt(0.0))
