# Geospatial Health & Air Quality Analysis Web App 
This project is an interactive web application built with [Solara](https://solara.dev/) for analyzing the impact of air pollution on different demographic groups in Helsinki.
The application allows users to upload their own health data, select an air pollutant, a date, and a specific age/health group. It then performs a geospatial analysis to correlate air quality data with population and health metrics.
## Geospatial-Challenge-Camp-2025_EurikaCollective
This repository contains the source code of the product made by team EurikaCollective for Geospatial Challenge Camp 2025. A solara based webapp for  Real time air quality monitoring and vulnerable community assessment in Helsinki region.

## Core Features

* **File Upload**: Users can upload custom health data in CSV format.
* **Interactive Controls**: A user-friendly sidebar allows for the selection of:
    * Date
    * Air Pollutant (`NO2`, `O3`, `PM10`, etc.)
    * Demographic Group (e.g., `70+ Resp`)
* **Geospatial Analysis**:
    * Merges demographic data with raster-based air quality data for the selected pollutant and date.
    * Categorizes geographic zones into risk levels (blue, orange, red) based on health data thresholds.
    * Saves the processed data as a GeoPackage file.
* **Interactive Map Visualization**:
    * Displays the results on an interactive Folium map.
    * Risk zones are color-coded, and layers can be toggled.
    * Tooltips show population data for each specific zone.
* **Data Download**: Users can download the resulting GeoPackage file containing the processed geospatial data.

## How It Works

1.  The user provides parameters (date, pollutant, age group) and optionally uploads health data.
2.  Upon clicking "Run Analysis", the script loads the corresponding air quality raster file and demographic vector data.
3.  It performs a spatial join to merge the two datasets, calculating the mean pollutant value for each demographic area.
4.  It uses the health data to create risk bins and classifies each area.
5.  When "Show Map" is clicked, the application generates and displays an interactive Folium map visualizing the population within the calculated risk zones.

## Key Libraries

* **`solara`**: For building the reactive web interface.
* **`geopandas`** & **`pandas`**: For vector data manipulation and analysis.
* **`rasterio`** & **`rioxarray`**: For reading and processing raster (air quality) data.
* **`folium`**: For creating the interactive map visualizations.
* **`shapely`**: For handling geometric objects and operations.
