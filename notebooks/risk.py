import geopandas as gpd
import streamlit as st
import numpy as np
import pydeck as pdk
import pandas as pd

st.set_page_config(
	layout="wide",
    page_title="Waste Site Explorer")


data = gpd.read_file('../data/site_metadata/full_v0_metadata.geojson')
data['lon'] = [point.x for point in data['geometry']]
data['lat'] = [point.y for point in data['geometry']]

custom_filter = st.sidebar.multiselect("Add custom parameter filter", data)

filter_index = np.array([False for _ in range(len(data))])

def filter_data(data, category, filter_val, filter_type):
    filter_index = np.array([False for _ in range(len(data))])
    if filter_type == ">":
        filter_index[data[category].astype(float) > filter_val] = True
    if filter_type == "<":
        filter_index[data[category].astype(float) < filter_val] = True
    if filter_type == "=":
        filter_index[data[category].astype(float) == filter_val] = True
    return filter_index

filter_indices = []
if len(custom_filter) > 0:
    for category in custom_filter:
        if np.max(data[category].astype(float)) <= 1:
            filter_val = st.sidebar.slider(category, 
                                np.min(data[category].astype(float)), np.max(data[category].astype(float)), 
                                0.0, step=0.001, format="%.3f")
        else:
            filter_val = st.sidebar.slider(category, 
                                np.min(data[category].astype(float)), np.max(data[category].astype(float)), 
                                0.0)
        filter_type = st.sidebar.radio('Comparison Type', [">", "<", "="], key=category)
        filter_indices.append(filter_data(data, category, filter_val, filter_type))

filter_index = np.array([False for _ in range(len(data))])
for i in range(len(data)):
    vals = [index[i] for index in filter_indices]
    if all(val == True for val in vals) == True:
        filter_index[i] = True

filtered_df = data[filter_index]

st.metric('Number of Sites', len(filtered_df))

map_chart = pdk.Deck(
    map_style='mapbox://styles/mapbox/satellite-v9',
    initial_view_state=pdk.ViewState(
         latitude=-3.2857382831459225,
         longitude=112.23373409248943,
         zoom=4,
         pitch=0,
         height=800
     ),
    layers = pdk.Layer(
             'ScatterplotLayer',
             data=filtered_df,
             get_position='[lon, lat]',
             get_color='[255, 206, 0]',
             radius_min_pixels=4,
             stroked=False,
             filled=True
         )
)
st.pydeck_chart(map_chart)