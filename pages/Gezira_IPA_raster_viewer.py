import streamlit as st
import rasterio
import xarray as xr
import pandas as pd
import altair as alt
import folium
from folium.raster_layers import ImageOverlay, TileLayer
from branca.colormap import LinearColormap
from streamlit_folium import folium_static
import numpy as np
from pyproj import Transformer
import re
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#===================================
# Page configuration
st.set_page_config(
    page_title="Gezira Irrigation Scheme Irrigation Performance Indicators by Sections Dashboard",
    page_icon="📈🌿",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# CSS styling
st.markdown("""
<style>
.reportview-container .css-1lcbmhc .css-1outpf7 {{
    padding-top: 35px;
}}
.reportview-container .main .block-container {{
    {max_width_str}
    padding-top: {0}rem;
    padding-right: {1}rem;
    padding-left: {1}rem;
    padding-bottom: {1}rem;
}}

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #1c1b1b;
    text-align: center;
    padding: 2px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}
            
img[data-testid="stLogo"] {
            height: 4.5rem;
}
</style>
""", unsafe_allow_html=True)


hide_github_icon = """
<style>
.css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK{ display: none; } #MainMenu{ visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
</style>
"""
st.markdown(hide_github_icon, unsafe_allow_html=True)
#========================================

# Access wapor via google bucket 
# dfm.columns = [x.replace('_', ' ') for x in dfm.columns]
logo_wide = r'data/logo_wide.png'
logo_small = r'data/logo_small.png'
ipa_ds_path = r'data/nc/IPA_results_Gezira.nc'


@st.cache_data(ttl=300)
def load_image(image_name: str) -> Image:
    """Displays an image.

    Parameters
    ----------
    image_name : str
        Local path of the image.

    Returns
    -------
    Image
        Image to be displayed.
    """
    return Image.open(image_name)

#######################

IPA_description = {
    "beneficial fraction": ":blue[Beneficial fraction (BF)] is the ratio of the water that is consumed as transpiration\
         compared to overall field water consumption (ETa). ${\\footnotesize BF = T_a/ET_a}$. \
         It is a measure of the efficiency of on farm water and agronomic practices in use of water for crop growth.",
    "crop water deficit": ":blue[crop water deficit (CWD)] is measure of adequacy and calculated as the ration of seasonal\
        evapotranspiration to potential or reference evapotranspiration ${\\footnotesize CWD= ET_a/ET_p}$",
    "relative water deficit": ":blue[relative water deficit (RWD)] is also a measure of adequacy which is 1 minus crop water\
          deficit ${\\footnotesize RWD= 1-ET_a/ET_p}$",
    "total seasonal biomass production": ":blue[total seasonal biomass production (TBP)] is total biomass produced in tons. \
        ${\\footnotesize TBP = (NPP * 22.222) / 1000}$",
    "seasonal yield": ":blue[seasonal yield] is the yield in a season which is crop specific and calculated using \
        the TBP and yield factors such as moisture content, harvest index, light use efficiency correction \
            factor and above ground over total biomass production ratio (AOT) \
                ${\\footnotesize Yiled = TBP*HI*AOT*f_c/(1-MC)}$",
    "crop water productivity": ":blue[crop water productivity (CWP)] is the seasonal yield per the amount of water \
        consumed in ${kg/m^3}$"
}

units = {'beneficial fraction':'-', 'crop water deficit': '-',
       'relative water deficit': '-', 'total seasonal biomass production': 'ton',
       'seasonal yield': 'ton/ha', 'crop water productivity': 'kg/m<sup>3</sup>'}

crop_calendar = {'wheats': 'November to March', 'sorgums':'June to December', 'cottons':'June to March'}
# Sidebar
with st.sidebar:

    st.logo(load_image(logo_wide), size="large", link='https://www.un-ihe.org/', icon_image=load_image(logo_small))
    st.title('Gezira Irrigation Performance Indicators')

    dfm = pd.read_csv(fr'data/Gezira_IPA_statistic_sorgums.csv')
    season_list = list(dfm.season.unique())[::-1]    
    selected_season = st.selectbox('Select a season', season_list)
    season_end_yr = selected_season.split('-')[1]    


    ll = list(dfm.columns.unique())[3:][::-1]

    indicator_lst = [' '.join(l.split('_')[1:]) for l in ll]
     
    indicator = st.selectbox('Select an indicator', set(indicator_lst))
    data_array_name = indicator
  

@st.cache_data
def read_raster_local(raster_path):
    with rasterio.open(raster_path) as dataset:  
        # dataset.nodata = -9999 
        data = dataset.read(1)
        transform = dataset.transform
        crs = dataset.crs
        nodata = -9999 #dataset.nodata
        bounds = dataset.bounds
    
    return data, transform, crs, nodata, bounds

@st.cache_data
def read_dataset(ds_path):
    with xr.open_dataset(ds_path) as dataset:  
        # data = dataset.beneficial_fraction[0].values
        transform = dataset.rio.transform()
        crs = dataset.rio.crs
        nodata = -9999 #dataset.nodata
        bd = dataset.rio.bounds()
        bounds = rasterio.coords.BoundingBox(bd[0], bd[1], bd[2], bd[3])
    
    return dataset, transform, crs, nodata, bounds


def display_image(data, transform, crs, variable, nodata, bounds):
    try:
        # Apply scale factor
        scale_factor = 1#0.001
        data = np.nan_to_num(data, nan=-9999)
        data = np.flip(data,0)
        data = data.astype(float)
        
        # Create a custom colormap
        colors = ['red', 'yellow', 'green']
        n_bins = 100
        cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
        
        # Normalize data
        valid_data = data[data != nodata]
        if valid_data.size == 0:
            st.error("No valid data available for the selected image.")
            return
        
        # vmin, vmax =  valid_data.min(), valid_data.max()#np.percentile(valid_data, [2, 98])
        # vmin = max(vmin, 0.001)
        vmin, vmax =  valid_data.min(), valid_data.max()#np.percentile(valid_data, [2, 98])
        # vmin = max(vmin, 0.001)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        
        # Apply colormap to data
        colored_data = cmap(norm(data))
        
        # Set alpha channel to 0 for no-data values
        colored_data[..., 3] = np.where(data == nodata, 0, 0.7)
        
        
        # Convert to PIL Image
        img = Image.fromarray((colored_data * 255).astype(np.uint8))
        
        # Save image to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Encode image to base64
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode()
    
        # Transform bounds to EPSG:4326 (WGS84)
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        left, bottom = transformer.transform(bounds.left, bounds.bottom)
        right, top = transformer.transform(bounds.right, bounds.top)

        # Create a folium map
        m = folium.Map(location=[(bottom + top) / 2, (left + right) / 2], zoom_start=8.3, control_scale=True)
        
        # Add aerial background
        TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Aerial Imagery',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Add the image overlay
        ImageOverlay(
            image=f"data:image/png;base64,{img_base64}",
            bounds=[[bottom, left], [top, right]],
            opacity=1.0,
            name=variable
        ).add_to(m)
    
        # Add the color map legend
        colormap = LinearColormap(colors=colors, vmin=vmin, vmax=vmax)
        colormap.add_to(m)
        colormap.caption = f"{variable} Values"

        
        # Add layer control and fullscreen option
        folium.LayerControl().add_to(m)
        folium.plugins.Fullscreen().add_to(m)
        
        # Display the map
        folium_static(m)
        
        # Display additional information
        # st.write(f"{variable} values range from {vmin:.2f} to {vmax:.2f} {units[indicator]}")
        
    except Exception as e:
        st.error(f"An error occurred while processing the data: {str(e)}")
        st.write("Debug information:")
        st.write(f"Data shape: {data.shape}")
        st.write(f"Transform: {transform}")
        st.write(f"CRS: {crs}")
        st.write(f"Bounds: {bounds}")
# @st.cache_data
def get_stats(_data):
      # Compute spatial statistics
    _data = _data.where(_data>0, np.nan)
    stats = {
        'Minimum': _data.min(dim=['latitude', 'longitude']),
        'Maximum': _data.max(dim=['latitude', 'longitude']),
        'Mean': _data.mean(dim=['latitude', 'longitude']),
        'Median': _data.median(dim=['latitude', 'longitude']),
        'St. deviation': _data.std(dim=['latitude', 'longitude']),
        "25% quantile": _data.quantile(0.25, dim=['latitude', 'longitude'], method='linear')
                        .drop_vars('quantile'),
        "75% quantile": _data.quantile(0.75, dim=['latitude', 'longitude'], method='linear')
                        .drop_vars('quantile'),
    }

    # pd.DataFrame.from_dict(d)
    df_stat = pd.DataFrame.from_dict({k: v.values.item() for k, v in stats.items()}, 
                                    orient='index', columns = ['Values']).round(2)
    df_stat.index.names = ['Stats']
    return df_stat

def main():

    try:
        variable = indicator.replace(' ', '_')
        # st.markdown(f"#### {indicator} [{units[indicator]}]")
        slected_time = f'{season_end_yr}-03-31'
        ds, transform, crs, nodata, bounds = read_dataset(ipa_ds_path)
        data =  ds.sel(time=slected_time)[variable]
        
        df_stats = get_stats(data)
    
        # st.title("### Gezira IPA RAster Viewer")
        col = st.columns((6.0, 2.0), gap='medium')
        with col[0]:
            st.markdown(f"#### Gezira IPA Raster Viewer: :blue[{indicator} [{units[indicator]}]] for :blue[{selected_season}] season")
            with st.spinner("Loading and processing data..."):
                display_image(data, transform, crs, variable, nodata, bounds)

        with col[1]:
            st.write('')
            st.markdown(f"##### :blue[Stats of {indicator} [{units[indicator]}]]")
            st.dataframe(df_stats, use_container_width=True)
            # st.markdown(df_stats.to_html(escape=False),unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred while processing the data: {str(e)}")
                        
            
if __name__ == "__main__":
    main()
