"""
Supernova Light Curve Viewer
Interactive visualization of synthetic photometry with adjustable parameters
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add parent directory to path (handle both absolute and relative paths)
SCRIPT_DIR = Path(__file__).parent.resolve()
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

# Change working directory to parent for imports
os.chdir(str(PARENT_DIR))

from core.utils import leer_spec, Syntetic_photometry_v2
from core.correction import correct_redeening

# Configure page
st.set_page_config(
    page_title="SN Light Curve Viewer",
    page_icon="✨",
    layout="wide"
)

# Matplotlib style for academic papers
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Constants
BASE_DIR = PARENT_DIR
DATA_DIR = BASE_DIR / "data"

# Rutas auto-detectadas por OS desde config
import sys
sys.path.insert(0, str(PARENT_DIR))
from config import RESPONSE_FOLDER
try:
    RESPONSE_DIR = Path(RESPONSE_FOLDER).resolve()
    if not RESPONSE_DIR.exists():
        st.warning(f"⚠️ Directorio de respuestas no encontrado: {RESPONSE_DIR}")
        RESPONSE_DIR = None
except Exception as e:
    st.warning(f"⚠️ Error al acceder al directorio de respuestas: {e}")
    RESPONSE_DIR = None

FILTER_CONSTANTS = {
    'U': 1.64e-08, 'B': 4.04e-08, 'V': 3.55e-08,
    'R': 2.18e-08, 'I': 1.16e-08,
    'u': 9.27e-09, 'g': 5.16e-08, 'r': 2.75e-08,
    'i': 1.39e-08, 'z': 8.68e-09
}

RESPONSE_FILES = {
    'U': 'spline_U.txt', 'B': 'spline_B.txt', 'V': 'spline_V.txt',
    'R': 'bessell_R_ph_lines.dat', 'I': 'bessell_I_ph_lines.dat',
    'u': "spline_u'.txt", 'g': "spline_g'.txt", 'r': "spline_r'.txt",
    'i': "spline_i'.txt", 'z': "spline_z'.txt"
}

FILTER_COLORS = {
    'U': '#9467bd', 'B': '#1f77b4', 'V': '#17becf',
    'R': '#ff7f0e', 'I': '#e377c2',
    'u': '#8c564b', 'g': '#2ca02c', 'r': '#d62728',
    'i': '#bcbd22', 'z': '#7f7f7f'
}

@st.cache_data
def load_response_curves():
    """Load filter response curves"""
    responses = {}
    if RESPONSE_DIR is None:
        st.error("❌ No se puede cargar las curvas de respuesta: directorio no disponible")
        return responses
    
    for filt, fname in RESPONSE_FILES.items():
        try:
            fpath = RESPONSE_DIR / fname
            if fpath.exists():
                df = pd.read_csv(fpath, sep=r'\s+', names=['wave', 'response'], comment='#')
                responses[filt] = df
            else:
                st.warning(f"⚠️ Archivo no encontrado para filtro {filt}: {fpath}")
        except Exception as e:
            st.warning(f"⚠️ Error al cargar filtro {filt}: {e}")
            continue
    return responses

@st.cache_data
def scan_supernova_files():
    """Scan for available supernova template files"""
    sn_files = {}
    for sn_type in ['Ia', 'Ibc', 'II', 'Ibc_old']:
        type_dir = DATA_DIR / sn_type
        if type_dir.exists():
            files = sorted(list(type_dir.glob('*.dat')))
            sn_files[sn_type] = [f.name for f in files]
    return sn_files

def generate_synthetic_photometry(spectra, phases, filters, responses, overlap_threshold=0.95,
                                 noise_level=0.15):
    """Generate synthetic photometry for selected filters (same as main_multiband.py, without Loess)"""
    results = {}
    
    for filt in filters:
        if filt not in responses:
            continue
            
        response_df = responses[filt]
        
        # Fotometría sintética con overlap threshold
        fases_lc = []
        fluxes_lc = []
        overlaps_lc = []
        
        for spec, fase in zip(spectra, phases):
            flux, overlap = Syntetic_photometry_v2(
                spec['wave'].values, spec['flux'].values,
                response_df['wave'].values, response_df['response'].values
            )
            if overlap > overlap_threshold:
                fases_lc.append(fase)
                fluxes_lc.append(flux)
                overlaps_lc.append(overlap)
        
        if len(fases_lc) == 0:
            continue
        
        lc_df = pd.DataFrame({
            'fase': fases_lc,
            'flux': fluxes_lc,
            'overlap': overlaps_lc
        }).sort_values('fase')
        
        # Calibración fotométrica
        mul = FILTER_CONSTANTS[filt]
        flux_calibrated = np.array(lc_df['flux']) / mul
        mag = -2.5 * np.log10(np.clip(flux_calibrated, 1e-20, None))
        
        # Aplicación de ruido
        flux_from_mag = 10 ** (-0.4 * mag)
        minimo_flux = np.min(flux_from_mag)
        flux_norm = flux_from_mag / minimo_flux
        
        flux_noisy_norm = np.random.normal(
            loc=flux_norm, scale=np.sqrt(np.abs(flux_norm)) * noise_level
        )
        
        flux_noisy = flux_noisy_norm * minimo_flux
        flux_noisy = np.clip(flux_noisy, 1e-20, None)
        mag_noisy = -2.5 * np.log10(flux_noisy)
        
        # Guardar resultados
        results[filt] = {
            'mjd': np.array(lc_df['fase']),
            'mag': mag,
            'mag_noisy': mag_noisy,
            'overlap': np.array(lc_df['overlap'])
        }
    
    return results

def plot_light_curves(lc_data, sn_name, z, ebv_host, ebv_mw, show_noisy=True):
    """Create academic-style multi-band light curve plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for filt, data in lc_data.items():
        color = FILTER_COLORS.get(filt, 'gray')
        
        if show_noisy:
            # Plot noisy data with line
            ax.plot(data['mjd'], data['mag_noisy'], 'o-', color=color,
                   markersize=5, linewidth=1.5, alpha=0.8,
                   label=f'{filt}', zorder=3)
        else:
            # Plot clean data
            ax.plot(data['mjd'], data['mag'], 'o-', color=color,
                   markersize=5, linewidth=1.5, alpha=0.8,
                   label=f'{filt}', zorder=3)
    
    ax.invert_yaxis()
    ax.set_xlabel('MJD (days)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Apparent Magnitude', fontsize=11, fontweight='bold')
    
    title = f'{sn_name}\n'
    title += f'z={z:.4f}, E(B-V)$_{{host}}$={ebv_host:.3f}, E(B-V)$_{{MW}}$={ebv_mw:.3f}'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', ncol=2, framealpha=0.9, fontsize=9)
    
    plt.tight_layout()
    return fig

# Main App
st.title("✨ Supernova Light Curve Viewer")
st.markdown("Interactive visualization of synthetic photometry with adjustable astrophysical parameters")

# Sidebar: File Selection
st.sidebar.header("1. Supernova Selection")
sn_files = scan_supernova_files()

sn_type = st.sidebar.selectbox("SN Type", list(sn_files.keys()))
sn_name = st.sidebar.selectbox("SN Template", sn_files[sn_type])

# Sidebar: Astrophysical Parameters
st.sidebar.header("2. Astrophysical Parameters")
z = st.sidebar.slider("Redshift (z)", 0.0, 0.1, 0.02, 0.001, 
                      help="Cosmological redshift")
ebv_host = st.sidebar.slider("E(B-V) Host", 0.0, 1.0, 0.1, 0.01,
                             help="Host galaxy extinction")
ebv_mw = st.sidebar.slider("E(B-V) Milky Way", 0.0, 0.5, 0.05, 0.01,
                           help="Galactic extinction")

# Sidebar: Filter Selection
st.sidebar.header("3. Filter Selection")
responses = load_response_curves()
available_filters = list(responses.keys())

default_filters = ['g', 'r', 'i'] if all(f in available_filters for f in ['g', 'r', 'i']) else available_filters[:3]
selected_filters = st.sidebar.multiselect(
    "Filters", available_filters, default=default_filters,
    help="Select photometric filters"
)

overlap_threshold = st.sidebar.slider("Overlap Threshold (%)", 80, 100, 95, 1) / 100.0

# Advanced options
with st.sidebar.expander("⚙️ Advanced Options"):
    noise_level = st.slider("Noise Level", 0.0, 0.3, 0.15, 0.01)
    show_noisy = st.checkbox("Show Noisy Data", value=True,
                            help="If unchecked, shows clean synthetic photometry")

# Load and process
if selected_filters:
    with st.spinner("Loading spectrum and generating photometry..."):
        # Load spectrum
        spec_path = DATA_DIR / sn_type / sn_name
        spectra, phases = leer_spec(spec_path, ot=False, as_pandas=True)
        
        # Apply corrections only if needed
        if z == 0 and ebv_host == 0 and ebv_mw == 0:
            # No corrections needed, use original spectra
            spectra_corr = spectra
            phases_corr = phases
            st.info("Using original spectra (no corrections applied)")
        else:
            # Apply corrections
            spectra_corr, phases_corr = correct_redeening(
                sn=sn_name.replace('.dat', ''),
                ESPECTRO=spectra,
                fases=phases,
                z=z if z > 0 else 0.001,  # Avoid z=0 issues
                ebmv_host=ebv_host,
                ebmv_mw=ebv_mw,
                reverse=True,
                use_DL=True
            )
        
        # Generate photometry
        lc_data = generate_synthetic_photometry(
            spectra_corr, phases_corr, selected_filters, 
            responses, overlap_threshold, noise_level=noise_level
        )
        
        # Display results
        if lc_data:
            st.success(f"Generated light curves for {len(lc_data)} filters")
            
            # Show plot
            fig = plot_light_curves(lc_data, sn_name.replace('.dat', ''), 
                                   z, ebv_host, ebv_mw, show_noisy)
            st.pyplot(fig)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Filters", len(lc_data))
            with col2:
                total_points = sum(len(data['mjd']) for data in lc_data.values())
                st.metric("Total Data Points", total_points)
            with col3:
                mjd_range = max(data['mjd'].max() for data in lc_data.values()) - \
                           min(data['mjd'].min() for data in lc_data.values())
                st.metric("MJD Coverage (days)", f"{mjd_range:.1f}")
            
            # Detailed table
            with st.expander("📊 Detailed Data per Filter"):
                for filt, data in lc_data.items():
                    st.subheader(f"Filter {filt}")
                    df_display = pd.DataFrame({
                        'MJD': data['mjd'],
                        'Magnitude (clean)': data['mag'],
                        'Magnitude (noisy)': data['mag_noisy'],
                        'Overlap': data['overlap']
                    })
                    st.dataframe(df_display, use_container_width=True)
        else:
            st.error("No valid photometry generated. Try adjusting the overlap threshold or selecting different filters.")
else:
    st.warning("Please select at least one filter")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Made for academic research**")
st.sidebar.markdown("ZTF Supernova Projection Pipeline")
