import geopandas as gpd
import pandas as pd
import networkx as nx
import osmnx as ox
import shapely
import numpy as np
import folium
import shapely
import h3
import requests
import os, glob
import dask.dataframe as dd
import dask_geopandas
from shapely.geometry import Point, Polygon 
import matplotlib.pyplot as plt

from ipyleaflet import Map, GeoJSON, GeoData, LayersControl, Marker, MarkerCluster, basemaps, Choropleth
from branca.colormap import linear
import ipywidgets as widgets

import solara
import solara.lab
import datetime as dt
#github_url = solara.util.github_url(__file__)
import rasterio as rio
from rasterio.plot import show
#from geotiff import GeoTiff
from pyproj import Transformer
import netCDF4
import xarray as xr
import rioxarray as xrr
import time
from geocube.vector import vectorize
from geocube.api.core import make_geocube


text = solara.reactive("Scandic Grand Central, Helsinki")
#date = solara.reactive(dt.date.today())
continuous_update = solara.reactive(False)
#global date
#date = dt.date.today()


@solara.component
def ClickButton():
    clicks, set_clicks = solara.use_state(0)

def create_button_click(val):
    def button_click():
        print(val)
    return button_click

def handle_interaction(**kwargs):
    #print(kwargs)
    latlon = kwargs.get("coordinates")
    if kwargs.get("type") == "click":
        output = widgets.Output()
        print(output)
        #widget.children = [label, output]
        with output:
            print(latlon)

def load_file_df(file):
    df = pd.read_csv(file["file_obj"])
    print("Loaded dataframe:")
    print(df)
    df.to_csv('./AQ data/Health_data/uploaded_file_health_data.csv',index=False)
    


def show_vector_map():
    print("inside show vector map!")
    demo = gpd.read_file("./AQ data/to_download/demo_aqi_health.gpkg")
    
    import matplotlib.colors as mcolors

    # Assume `demo` is your GeoDataFrame with ('OBJECTID', 'geometry', '70+ Resp', 'colour')
    # Compute map center from data bounds
    minx, miny, maxx, maxy = demo.total_bounds
    center = [(miny + maxy) / 2, (minx + maxx) / 2]

    # Precompute normalization bounds
    vmin = demo["population"].min()
    vmax = demo["population"].max()

    def style_function(feature):
        #print("inside style function!")
            # Precompute normalization bounds
        
        base = feature["properties"]["bins"]                 # matplotlib color code
        val  = feature["properties"]["population"]               # response value
        # Normalize between 0 and 1
        norm = (val - vmin) / (vmax - vmin)                     # min-max normalization :contentReference[oaicite:3]{index=3}
        rgba = np.array(mcolors.to_rgba(base))                 # get RGBA tuple :contentReference[oaicite:4]{index=4}
        # Interpolate each RGB toward white (1,1,1) by (1 - norm)
        rgb  = rgba[:3] * norm + np.array([1, 1, 1]) * (1 - norm)
        hexc = mcolors.to_hex(rgb)                             # back to HEX :contentReference[oaicite:5]{index=5}
        return {
            "fillColor": hexc,
            "fillOpacity": .4,
            "color": "#000000",
            "weight": 0.01,
            "stroke":      False,
        }

    # Build the map
    m = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap",    # Folium’s Mapnik tiles :contentReference[oaicite:1]{index=1}
        attr="&copy; OpenStreetMap contributors")             # center on data :contentReference[oaicite:6]{index=6}

    # Add the styled GeoJson layer
    folium.GeoJson(
        demo.__geo_interface__, 
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["OBJECTID", "population"],
            aliases=["ID", "population"],
            localize=True
        )                                                       # interactive tooltips :contentReference[oaicite:7]{index=7}
    ).add_to(m)
    display(m)
    

 
def show_raster_map(img_path):
    print("inside show raster map")
    ## LC08 RGB Image
    in_path = img_path

    dst_crs = 'EPSG:4326'

    with rio.open(in_path) as src:
        
        img = src.read()
            
        src_crs = src.crs['init'].upper()
        min_lon, min_lat, max_lon, max_lat = src.bounds
        
    ## Conversion from UTM to WGS84 CRS
    bounds_orig = [[min_lat, min_lon], [max_lat, max_lon]]

    bounds_fin = []
     
    for item in bounds_orig:   
        #converting to lat/lon
        lat = item[0]
        lon = item[1]
        proj = Transformer.from_crs(int(src_crs.split(":")[1]), int(dst_crs.split(":")[1]), always_xy=True)
        lon_n, lat_n = proj.transform(lon, lat)  
        bounds_fin.append([lat_n, lon_n])

    # Finding the centre latitude & longitude    
    centre_lon = bounds_fin[0][1] + (bounds_fin[1][1] - bounds_fin[0][1])/2
    centre_lat = bounds_fin[0][0] + (bounds_fin[1][0] - bounds_fin[0][0])/2

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start = 11) # tiles='Stamen Terrain',

    # Overlay raster (RGB) called img using add_child() function (opacity and bounding box set)
    m.add_child(folium.raster_layers.ImageOverlay(img.transpose(1, 2, 0),
                                    opacity=.7, 
                                     bounds = bounds_fin))

    # Display map 
    display(m)
    print("Exiting show raster map")



def show_final_map_data(date):
    try:
        #results_df.rio.to_raster("./AQ data/temp/demo_aqi_GB_gdf_cog.tif", driver="COG", compress="LZW")
        show_raster_map("./AQ data/temp/demo_aqi_GB_gdf_cog.tif")

    except:
        print("show_final_map_data failed!!")
        
   
def show_location_map(address):
    try:
        lat, lon = ox.geocode(address)
        poi = gpd.GeoDataFrame({"geometry": [Point(lon, lat)], "name": address, "id": [0]}, index=[0], crs="epsg:4326")
        center = (lat, lon)
        zoom = 11
        # Create the map
        m = Map(center=center, zoom=zoom,scroll_wheel_zoom= True)
        markers = [Marker(location=(point.y, point.x)) for point in poi.geometry]

        #for clickable_marker in markers:
        #    clickable_marker.on_click(create_button_click(address))
        # Create and add MarkerCluster to the map
        marker_cluster = MarkerCluster(markers=markers, name='Cluster Markers')
        # Display the map
        m.add_layer(marker_cluster)
            # Add layer control
        m.add_control(LayersControl())
        #label = widgets.Label('Clicked location')
       # widget = widgets.VBox([label])
        #control = ipyleaflet.WidgetControl(widget=widget, position='bottomright')
        #m.add_control(control)
        m.on_interaction(handle_interaction)
        display(m)
        #plotly map? 
    except:
        address,lat,lon="Not a valid address", "-", "-"
                
    solara.Markdown(f"**Entered address with lat and lon**: {address, lat, lon}")

def show_selected_date(date):
    try:
        file2read = netCDF4.Dataset('./AQ data/enfuser_helsinki_metropolitan_20250324T180000_20250325T080000_20250325T110000_latlon.nc','r')
        print(file2read.variables.keys())
        health_data_df = pd.read_csv('./AQ data/Health_data/*.csv')
        print(health_data_df)

        print('File read sucess!')
        solara.Markdown(f"**Entered date**: {str(date.value)}")
    except:
        #date=dt.date.today()
        #solara.Markdown(f"**Entered date**: {str(date.value)}")
        #Page()
        return
     
def merge_AQdata(date,selected_AQI,selected_AGC):
#    demo_data = gpd.read_file('./AQ data/demo data helsinki pack.gpkg')
#    date_yymmdd = str.split(str(date.value), sep='-')
#    raster_to_read = './AQ data/' + date_yymmdd[0] + date_yymmdd[1]+ date_yymmdd[2] + '_' +str(selected_AQI) + '_avg.tif'  
#    print(raster_to_read)

    try:
        #ds = xr.open_dataset('./AQ data/enfuser_helsinki_metropolitan_20250324T180000_20250325T080000_20250325T110000_latlon.nc')
        #ncdf = ds.to_dataframe()
        demo_data = gpd.read_file('./AQ data/demo data helsinki pack_columns.gpkg')

        date_yymmdd = str.split(str(date.value), sep='-')
        
        raster_to_read = './AQ data/' + date_yymmdd[0] + date_yymmdd[1]+ date_yymmdd[2] + '_' +str(selected_AQI) + '_avg.tif'  

        dataarray = xrr.open_rasterio(raster_to_read)
        #dataarray = dataarray.rename({"band": selected_AQI})
        gdf = vectorize(dataarray.astype("float32"))
        aqi_gdf = gdf.rename(columns={"_data": selected_AQI})
        demo_aqi = demo_data.sjoin(aqi_gdf, how="inner")
        demo_aqi_GB = demo_aqi.groupby(['OBJECTID', 'NRO', 'euref_x', 'euref_y', '0-9', '0-9 No resp',
       '0-9 Resp', '10 - 69', '10-69 No resp', '10-69 Resp', '70+',
       '70+ No Resp', '70+ Resp', 'area', 'perimeter', 'geometry'])[selected_AQI].agg('mean').reset_index()

        demo_aqi_GB_gdf = gpd.GeoDataFrame(demo_aqi_GB)
        


        #filtered_health_data = health_data[health_data.columns[health_data.columns.str.contains(selected_AQI)]]
        #print(filtered_health_data)
        print(dataarray)
       
        
        #health_data_df = pd.read_csv('./AQ data/Health_data/uploaded_file_health_data.csv',index_col="Unnamed: 0") #
        health_data_df = pd.read_csv('./AQ data/Health_data/GCC_Risk assessment_new.csv',index_col=0) #
        
        
        health_df_temp = health_data_df[health_data_df.columns[health_data_df.columns.str.contains(str(selected_AQI))]].copy()
        print(selected_AGC)
         
        labels = ['blue','orange','red']
        bins   = [0] + health_df_temp.loc[selected_AGC].values.tolist()
        processed_df = demo_aqi_GB_gdf[['OBJECTID','geometry',selected_AGC,selected_AQI]].copy()
        
        print(health_data_df)

        processed_df["bins"] = pd.cut(processed_df[selected_AQI],bins=bins,labels=labels,include_lowest=True)

        
        #processed_df.to_csv("./AQ data/to_download/demo_aqi_health.csv")
        processed_df.rename(columns={selected_AGC:'population'},inplace=True)
        
        processed_df.to_file("./AQ data/to_download/demo_aqi_health.gpkg")
        print('File read sucess!')

        demo_aqi_GB_raster = make_geocube(vector_data=processed_df,
            measurements=["OBJECTID", 'population'], ## ,selected_AQI
            resolution=(-0.0025, 0.0025),
            output_crs="epsg:4326",)
        demo_aqi_GB_raster.rio.to_raster("./AQ data/temp/demo_aqi_GB_gdf_cog.tif", driver="COG", compress="LZW")

        ##save the final results to_download folder
        
        solara.Markdown(f"**Entered date**: {str(date.value)}")
    except:
        #date=dt.date.today()
        #solara.Markdown(f"**Entered date**: {str(date.value)}")
        #Page()
        #demo_aqi_GB_gdf = gpd.GeoDataFrame()
        print("Try function failed")
        
        return #demo_aqi_GB_gdf

#def merge_health_AQdata(date,selected_HealthGroup):
    #demo_aqi_GB_gdf.rio.to_raster("./AQ data/temp/demo_aqi_GB_gdf_cog.tif", driver="COG", compress="LZW")
        


        
def show_raster_map_data(date):
    if date:
        print("Date validated")
        img_path = "./AQ data/filtered_data.tif"
        show_raster_map(img_path)
    #show(rio.open(img_path))
    #geo_tiff = GeoTiff(img_path, crs_code=4326)
    #raster_image = folium.raster_layers.ImageOverlay(name="Helsinki_AQI",image = img_path, bounds=[[25, -180], [80, 180]],interactive=True,zindex=1,)
    #raster_image.add_to(m)
    else:
        return

def download_data(date):
    try:
        #ds = xr.open_dataset('./AQ data/enfuser_helsinki_metropolitan_20250324T180000_20250325T080000_20250325T110000_latlon.nc')
        #ncdf = ds.to_dataframe()
        import glob
        files = glob.glob('./AQ data/to_download/*.csv')
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df.to_csv("download.csv",index=False)
        #solara.Markdown(f"**Data download date**: {str(date.value)}")
        solara.Markdown(f"**Downloaded Successfully!**")
        print(df)
    except:
        solara.Markdown(f"**Download unsuccessful**")
        print("Not inside download")

@solara.component
def Page():
    with solara.AppBar():
        solara.lab.ThemeToggle()

        
    with solara.Column() as main:
        solara.Title("Air pollution data visualization")
        with solara.Sidebar():
            with solara.Card("Provide your health data"):
                with solara.Column():
                    #solara.Checkbox(label="Continuous update", value=continuous_update)
                    #solara.InputText("Enter some address in Helsinki", value=text, continuous_update=continuous_update.value) #update_events="keyup.enter")#
                    solara.FileDrop(label="Drop file to see dataframe!", on_file=load_file_df)
    
        solara.Title("First form")
        with solara.Sidebar():
            with solara.Card("Put date"):
                with solara.Column():
                    date = solara.use_reactive(dt.date.today())
                    print(date.value)
                    solara.lab.InputDate(date)
                    
            with solara.Card("Select a pollutant"):
                with solara.Column():      
                    option = solara.use_reactive("NO2")
                    options = ["NO2", "O3", "PM10", "PM2p5", "BC", "SO2", "TRS"]
                    solara.Select(label="Select pollutant:", value=option, values=options)
                    if option.value in ["NO2", "O3", "PM10", "PM2p5", "BC", "SO2", "TRS"]:
                        time.sleep(1) # Simulate a process
                        ##Call a function to display a particular pollutant results
                        selected_option = option.value
                        solara.Markdown(f"**Result for: {selected_option}**")

            with solara.Card("Select an age group and preconditions"):
                with solara.Column():      
                    option = solara.use_reactive("70+ Resp")
                    options = ['0-9 No resp', '0-9 Resp', '10-69 No resp', '10-69 Resp', '70+ No Resp','70+ Resp']
                    solara.Select(label="Select age group:", value=option, values=options)
                    if option.value in ['0-9 No resp', '0-9 Resp', '10-69 No resp', '10-69 Resp', '70+ No Resp','70+ Resp']:
                        time.sleep(1) # Simulate a process
                        ##Call a function to display a particular pollutant results
                        selected_option_ag = option.value
                        solara.Markdown(f"**Result for: {selected_option_ag}**")

            with solara.Card("Run Analysis "):
                with solara.Column():
                    clicked_analysis = solara.use_reactive(False)
                    solara.Button(label=f"Run analysis on the selected date", on_click=lambda: clicked_analysis.set(True) )
                    if clicked_analysis.value:
                        #show_selected_date(date)
                        merge_AQdata(date,selected_option,selected_option_ag) ### here we merge the uploaded file and AQ data
                        #solara.Text(str(date.value))
                        solara.Markdown(f"**Run analysis for {selected_option_ag} is completed! Click show map now to view results. **")
                        clicked_analysis.set(False)

                        
            with solara.Card("Download Results"):
                with solara.Column():
                    clicked2 = solara.use_reactive(False)
                    solara.Button(label=f"Download the data ", on_click=lambda: clicked2.set(True), icon_name="mdi-thumb-up")
                    if clicked2.value:
                        print("inside download")
                        download_data(date)
                        clicked2.set(False)
                        #solara.Text("Downloaded Successfully!")
                    #solara.FileDownload(data=download_data(date), filename="my_file.csv")
                    

    solara.Info("AQ Map of Helsinki")
    #show_location_map(text.value)
    clicked_final = solara.use_reactive(False)
    solara.Button(label=f"Show Map", on_click=lambda: clicked_final.set(True) )

    if clicked_final.value:
        print("Showing prelim map")
        try:
            show_vector_map()
        except:
            show_raster_map_data(date)
        #show_raster_map_data(date)
        #show_final_map_data(date)
    
#    if clicked_final.value and clicked_analysis.value:
        #show_raster_map_data(date)
        #try:
#        show_final_map_data(date)
#        clicked_analysis.set(False)
        #except:
            #print("Try showing results after analysis failed!")
        #show_raster_map_data(date)
        
#    elif clicked_final.value and not clicked_analysis.value:
#        print("Try showing results after analysis failed!")
#        show_raster_map_data(date)
        #clicked_final.set(False)

Page()    
    #return main
        #show_entered_date(solara.Text(str(date.value)))

        
        #solara.Button(label="View location in map",on_click= show_location_map(text.value),outlined=True)
       
#        with solara.Card("Use solara.Column to create a full width column"):
#            with solara.Column():
#                solara.Success("I'm first in this full with column")
#                solara.Warning("I'm second in this full with column")
#                solara.Error("I'm third in this full with column")



#@solara.component
#def Layout(children):
#    route, routes = solara.use_route()
#    return solara.AppLayout(children=children)
