"""
02_process_lulc_satellite.py
Multi-Temporal Satellite Classification & Impervious Surface Tracking (2001-2025)
Processes Landsat 5 TM, 7 ETM+, 8 OLI, and Sentinel-2 Multi-Spectral Imagery.
"""

def compute_spectral_indices(blue, green, red, nir, swir1):
    """
    Calculates essential indices for urban flood and land cover separation:
    - NDVI = (NIR - Red) / (NIR + Red)           [Vegetation Health]
    - NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)       [Built-up & Impervious]
    - MNDWI = (Green - SWIR1) / (Green + SWIR1)  [Open Water & Wetlands]
    - UI = (SWIR2 - NIR) / (SWIR2 + NIR)         [Urban Index]
    """
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndbi = (swir1 - nir) / (swir1 + nir + 1e-6)
    mndwi = (green - swir1) / (green + swir1 + 1e-6)
    return {'NDVI': ndvi, 'NDBI': ndbi, 'MNDWI': mndwi}

def classify_pixel(indices):
    """Rule-based decision tree for primary land-cover classification"""
    if indices['MNDWI'] > 0.15:
        return 'Water Body / Canal'
    elif indices['NDBI'] > 0.05 and indices['NDVI'] < 0.25:
        return 'Built-up / Impervious'
    elif indices['NDVI'] > 0.40:
        return 'Dense Vegetation'
    else:
        return 'Bare Soil / Lowland Fallow'

if __name__ == '__main__':
    print("[Remote Sensing] Satellite multi-spectral processing pipeline configured for Landsat/Sentinel-2.")
    sample_indices = compute_spectral_indices(0.08, 0.11, 0.14, 0.22, 0.32)
    print("  Extracted Sample Indices:", {k: round(v, 3) for k, v in sample_indices.items()})
    print("  Predicted Class:", classify_pixel(sample_indices))
