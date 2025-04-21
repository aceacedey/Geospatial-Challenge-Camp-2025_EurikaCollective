import geopandas as gpd
import pandas as pd
import folium
import solara
import solara.lab
import datetime as dt
import shapely
import numpy as np
import rasterio as rio
from rasterio.plot import show
from pyproj import Transformer
import time
import glob
from geocube.vector import vectorize
from geocube.api.core import make_geocube
from shapely.geometry import Point, Polygon
import matplotlib.colors as mcolors
import rioxarray as xrr
import xarray as xr
import io
import os
#from solara.lab import async_wrap_in_thread

def _load_csv(info, df_reactive):
    """
    info may be either a FileInfo-like object with .data
    or a dict with a 'data' key.
    """
    # extract raw bytes
    raw = getattr(info, "data", None) or info.get("data")
    # wrap in BytesIO and read
    try:
        new_df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        # maybe it's a text string?
        new_df = pd.read_csv(io.StringIO(raw.decode() if isinstance(raw, bytes) else raw))
    df_reactive.set(new_df)
    new_df.to_csv('./AQ data/Health_data/uploaded_file_health_data.csv', index=True)
    
def load_file_df(file):
    """Load and save uploaded health data CSV file"""
    df = pd.read_csv(file["file_obj"])
    print("Loaded dataframe:")
    print(df)
    df.to_csv('./AQ data/Health_data/uploaded_file_health_data.csv')


def show_vector_map(date,selected_AQI,selected_AGC):
    """Display interactive vector map with population data"""
    #demo = gpd.read_file("./AQ data/to_download/demo_aqi_health.gpkg")

    date_yymmdd = str(date.value).split('-')
    processed_data_path = f'./AQ data/Processed_data/{"".join(date_yymmdd)}_{selected_AQI}_{selected_AGC}_Helsinki.gpkg'

    demo = gpd.read_file(processed_data_path)
    
    minx, miny, maxx, maxy = demo.total_bounds
    center = [(miny + maxy) / 2, (minx + maxx) / 2]

    vmin = demo["population"].min()
    vmax = demo["population"].max()

    def style_function(feature):
        base = feature["properties"]["bins"]
        val = feature["properties"]["population"]
        norm = (val - vmin) / (vmax - vmin)
        rgba = np.array(mcolors.to_rgba(base))
        rgb = rgba[:3] * norm + np.array([1, 1, 1]) * (1 - norm)
        return {
            "fillColor": mcolors.to_hex(rgb),
            "fillOpacity": .8,
            "color": "#000000",
            "weight": 0.01,
            "stroke": False,
        }

    m = folium.Map(location=center, zoom_start=11,tiles= "CartoDB Positron",  attr='&copy; <a href="https://carto.com/attributions">CARTO</a> ' )
    folium.GeoJson(
        demo.__geo_interface__,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["OBJECTID", "population"],
            aliases=["ID", "Population"],
            localize=True
        )
    ).add_to(m)
    display(m)


def show_raster_map(img_path):
    """Display raster map from given image path"""
    dst_crs = 'EPSG:4326'

    with rio.open(img_path) as src:
        img = src.read()
        src_crs = src.crs['init'].upper()
        min_lon, min_lat, max_lon, max_lat = src.bounds

    bounds_fin = []
    for item in [[min_lat, min_lon], [max_lat, max_lon]]:
        proj = Transformer.from_crs(int(src_crs.split(":")[1]), 
                                  int(dst_crs.split(":")[1]), 
                                  always_xy=True)
        lon_n, lat_n = proj.transform(item[1], item[0])
        bounds_fin.append([lat_n, lon_n])

    centre_lon = (bounds_fin[0][1] + bounds_fin[1][1]) / 2
    centre_lat = (bounds_fin[0][0] + bounds_fin[1][0]) / 2

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=11)
    m.add_child(folium.raster_layers.ImageOverlay(
        img.transpose(1, 2, 0),
        opacity=.7,
        bounds=bounds_fin
    ))
    display(m)


def merge_AQdata(date, selected_AQI, selected_AGC):
    """Merge air quality data with health data"""
#    try:
    demo_data = gpd.read_file('./AQ data/demo data helsinki pack_columns.gpkg')
    date_yymmdd = str(date.value).split('-')
    raster_path = f'./AQ data/{"".join(date_yymmdd)}_{selected_AQI}_avg.tif'

    dataarray = xrr.open_rasterio(raster_path)
    gdf = vectorize(dataarray.astype("float32"))
    aqi_gdf = gdf.rename(columns={"_data": selected_AQI})
    demo_aqi = demo_data.sjoin(aqi_gdf, how="inner")

    demo_aqi_GB = demo_aqi.groupby([
        'OBJECTID', 'NRO', 'euref_x', 'euref_y', '0-9', '0-9 No resp',
        '0-9 Resp', '10 - 69', '10-69 No resp', '10-69 Resp', '70+',
        '70+ No Resp', '70+ Resp', 'area', 'perimeter', 'geometry'
    ])[selected_AQI].agg('mean').reset_index()

    demo_aqi_GB = gpd.GeoDataFrame(demo_aqi_GB)

    health_data_df = pd.read_csv(
        './AQ data/Health_data/GCC_Risk assessment_new.csv', 
        index_col=0
    )
    print("upto health data df")
    health_df_temp = health_data_df[health_data_df.columns[health_data_df.columns.str.contains(str(selected_AQI))]].copy()
    print(selected_AGC)
    
    labels = ['blue', 'orange', 'red']
    bins = [0] + health_df_temp.loc[selected_AGC].values.tolist()
    print("upto health data df temp")
    processed_df = demo_aqi_GB[['OBJECTID', 'geometry', selected_AGC, selected_AQI]].copy()
    processed_df["bins"] = pd.cut(processed_df[selected_AQI], bins=bins, labels=labels, include_lowest=True)
    processed_df.rename(columns={selected_AGC: 'population'}, inplace=True)
   
    processed_data_path = f'./AQ data/Processed_data/{"".join(date_yymmdd)}_{selected_AQI}_{selected_AGC}_Helsinki.gpkg'
    processed_df.to_file(processed_data_path)

    

    demo_aqi_GB_raster = make_geocube(
        vector_data=processed_df,
        measurements=["OBJECTID", 'population'],
        resolution=(-0.0025, 0.0025),
        output_crs="epsg:4326",
    )
    demo_aqi_GB_raster.rio.to_raster(
        "./AQ data/temp/demo_aqi_GB_gdf_cog.tif",
        driver="COG",
        compress="LZW"
    )

#    except Exception as e:
#        print(f"Data processing failed: {str(e)}")
#        solara.Markdown(f"Data processing failed: {str(e)}")

    return processed_df

def _run_analysis(date, pollutant, age_group, succeeded, did_run):
    """
    Calls merge_AQdata and stores its return in result_reactive.
    If it raises or returns falsy, we treat it as failure.
    
    """
    
    if not did_run.value:
            did_run.set(True)
    try:
        out = merge_AQdata(date, pollutant, age_group)
        succeeded.set(out is not None)   # or whatever “truthy” means for your func
    except Exception:
        succeeded.set(False)
    finally:
        did_run.set(True)

def run_analysis(date, pollutant, age_group):
    # Replace this with your actual analysis logic
    # For example, call your merge_AQdata function here
    print("inside run analysis")
    try:
        # Call your actual analysis function here
        result = merge_AQdata(date, pollutant, age_group)
        return result is not None
    except Exception as e:
        print(f"Analysis failed: {e}")
        return False



@solara.component
def Page():
    """Main application component"""
    with solara.AppBar():
        solara.lab.ThemeToggle()

    with solara.Column() as main:
        solara.Title("Effects of Air Pollution on Different Health and Age Groups based on Demographic Data")
        df = solara.use_reactive(None)
        show = solara.use_reactive(False)
        with solara.Sidebar():
        
            with solara.Card("Upload Health Data"):
                # 2) FileDrop sets the reactive df when a file is dropped
                solara.FileDrop(
                label="Drop CSV file",
                lazy=False,   # force .data to be loaded
                on_file=lambda info: _load_csv(info, df)
                )
                solara.Button(
                    label="Show Uploaded File",
                    icon_name="mdi-eye",
                    on_click=lambda: show.set(True),
                )

                 # 4) Only once the button’s been clicked AND df is loaded do we render
                if show.value and df.value is not None:
                    solara.DataFrame(df.value, items_per_page=10)

            with solara.Card("Parameters"):
                date = solara.use_reactive(dt.date.today())
                solara.lab.InputDate(date)
                
                pollutant = solara.use_reactive("NO2")
                solara.Select(
                    label="Select Pollutant:",
                    value=pollutant,
                    values=["NO2", "O3", "PM10", "PM2p5", "BC", "SO2", "TRS"]
                )
                
                age_group = solara.use_reactive("70+ Resp")
                solara.Select(
                    label="Select Age Group:",
                    value=age_group,
                    values=['0-9 No resp', '0-9 Resp', '10-69 No resp', 
                           '10-69 Resp', '70+ No Resp','70+ Resp']
                )

                # Reactive variable to track the status
                status = solara.use_reactive("idle")  # Possible values: 'idle', 'running', 'succeeded', 'failed'

                # Define the on_click handler
                def on_click():
                    if status.value != "running": #"idle":
                        status.set("running")
                        success = run_analysis(date, pollutant.value, age_group.value)
                        status.set("succeeded" if success else "failed")

                # Determine button label and color based on status
                if status.value == "idle":
                    button_label = "Run Analysis"
                    button_color = "primary"
                    button_disabled = False
                elif status.value == "running":
                    button_label = "Running..."
                    button_color = "warning"
                    button_disabled = True
                elif status.value == "succeeded":
                    button_label = "Run Analysis"
                    button_color = "primary"
                    button_disabled = False
                elif status.value == "failed":
                    button_label = "Run Analysis"
                    button_color = "failed"
                    button_disabled = False
                else:
                    button_label = "Run Analysis"
                    button_color = "primary"
                    button_disabled = False

                # UI Components
                with solara.Card("Analysis Controls"):
                    solara.Button(
                        label=button_label,
                        color=button_color,
                        disabled=button_disabled,
                        on_click=on_click
                    )
                # Display analysis result
                if status.value == "succeeded":
                    solara.Info(f"✅ Analysis succeeded! Click Show Map to view results for {age_group.value} against {pollutant.value}!")
                elif status.value == "failed":
                    solara.Markdown("❌ Analysis failed. Please check your inputs and try again.")
                    #print(success)
                #status.set("idle")
#                if did_run.value:
#                    if succeeded.value:
#                        solara.Info(f"✅ Analysis succeeded! Click Show Map to view results for {age_group.value} against {pollutant.value}!")
#                    else:
#                        solara.Markdown("❌ Analysis failed. Please check your inputs and try again.")
                    # reset so next click re‑runs
                     #   did_run.set(False)
                
#                if clicked_analysis.value:
#                    merge_AQdata(date, pollutant.value, age_group.value)
#                    solara.Info("Analysis completed! Click Show Map to view results.")
#                    solara.Markdown(f"**Analysis completed! Click Show Map to view results for {age_group.value} against {pollutant.value}!**")
#                    clicked_analysis.set(False)
                    

                date_yymmdd = str(date.value).split('-')
                processed_data_path = (
                    f"./AQ data/Processed_data/"
                    f"{''.join(date_yymmdd)}_{pollutant.value}_{age_group.value}_Helsinki.gpkg")

                # 2. Reactive flag to show the download link only after click
            show_download = solara.use_reactive(False)

            with solara.Card("Download Processed Data"):
                # 3. Button to activate the download component
                solara.Button(
                    label="Prepare Download",
                    icon_name="mdi-download",
                    on_click=lambda: show_download.set(True),
                )

                # 4. Conditionally render the FileDownload once button is clicked
                if show_download.value:
                    try:
                        # Open file in binary mode
                        file_obj = open(processed_data_path, "rb")
                        solara.FileDownload(
                            data=file_obj,
                            filename=os.path.basename(processed_data_path),
                            label="Click here to save the file",
                            icon_name="mdi-file-download",
                        )
                    except:
                        solara.Markdown(f"**❌ Analysis not ready for {age_group.value} against {pollutant.value} on {str(date.value)}.**")

        #solara.Info("Helsinki, Finland")
        show_map = solara.use_reactive(False)
        solara.Button(
            label="Show Map", 
            on_click=lambda: show_map.set(True),
            color="primary"
        )

        if show_map.value:
            try:
                show_vector_map(date, pollutant.value, age_group.value)
                solara.Info(f"✅ Visualization successful for {age_group.value} against {pollutant.value} on {str(date.value)}.")
            except Exception as e:
                print(f"Vector map failed due to invalid parameters: {str(e)}")
                solara.Markdown(f"**❌ Analysis not ready for {age_group.value} against {pollutant.value} on {str(date.value)}.**")
                show_raster_map("./AQ data/filtered_data.tif")

    return main


Page()
