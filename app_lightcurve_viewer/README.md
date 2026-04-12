# Supernova Light Curve Viewer

Interactive Streamlit app for visualizing supernova synthetic photometry with adjustable astrophysical parameters.

## Features

- 📂 Browse and select SN templates from data folder (Ia, Ibc, II)
- 🔭 Adjust redshift, host extinction, and Milky Way extinction in real-time
- 🎨 Multi-band photometry visualization (ZTF g/r/i, Johnson-Cousins U/B/V/R/I, SDSS u/g/r/i/z)
- 📊 Academic paper-ready plots with proper formatting
- 📈 Interactive parameter exploration

## Installation

Required packages:
```bash
pip install streamlit pandas numpy matplotlib
```

## Usage

From the `proyeccion` directory:

```bash
streamlit run app_lightcurve_viewer/lightcurve_viewer.py
```

Or from within the app directory:

```bash
cd app_lightcurve_viewer
streamlit run lightcurve_viewer.py
```

## Controls

**Left Sidebar:**
1. **SN Selection**: Choose type (Ia/Ibc/II) and template
2. **Astrophysical Parameters**:
   - Redshift (z): 0.0 - 0.1
   - E(B-V) Host: 0.0 - 1.0
   - E(B-V) MW: 0.0 - 0.5
3. **Filter Selection**: Pick photometric bands
4. **Overlap Threshold**: Minimum spectral coverage (80-100%)

**Main Panel:**
- Light curve plot (all selected filters)
- Summary statistics
- Detailed data tables

## Notes

- Light curves update automatically when parameters change
- Plots use academic paper formatting (serif fonts, proper axis labels)
- Data points require ≥95% overlap between spectrum and filter response by default
