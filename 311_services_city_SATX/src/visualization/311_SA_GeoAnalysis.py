# %% [markdown]
# <a href="https://colab.research.google.com/github/yassmin1/311_services_city_SATX/blob/main/311_SA_GeoAnalysis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Mapping

# %%
#!apt install python3-folium  # For folium and it's necessary to plot the gpd.explore
#!pip install descartes
#pip install contextily


# %%
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS, Transformer,Proj
import folium
from folium.plugins import HeatMap
import webbrowser
import os
import contextily as cx
from matplotlib.colors import rgb2hex,to_hex,ListedColormap,cnames
import seaborn as sns
from sklearn.utils import resample
import pandas as pd 
import matplotlib.pyplot as plt
#
input_file=r'C:\Users\Rayan\OneDrive\Documents\GitHub\311_services_city_SATX\311_services_city_SATX\src\data\311_clean.csv'


# %%
df=pd.read_csv(input_file)

# %%
df[['YCOORD','XCOORD']].head()

# %%


# Define CRS objects for UTM and WGS84 (used for latitude and longitude)
utm_crs = CRS.from_epsg( 3674  )#32614)# ET:3081)#26914)
wgs84_crs = CRS.from_epsg(4326 )#4326) 4152,4204 # EPSG code for WGS84
#wgs84_crs = CRS.from_epsg(3857)
print(wgs84_crs)
# Create a transformer object
transformer = Transformer.from_crs(utm_crs, wgs84_crs)
easting_coords=df['XCOORD']
northing_coords=df['YCOORD']

# Transform each coordinate pair (Easting, Northing) to (latitude, longitude)
lat_list = []
lon_list = []
for easting, northing in zip(easting_coords, northing_coords):
  lat, lon = transformer.transform(easting, northing)
  lat_list.append(lat)
  lon_list.append(lon)

df["lat"]=lat_list
df["lon"]=lon_list
df[['lat', 'lon','Category']]

# %% [markdown]
# ## output csv file

# %%
# save file with lot lon
# new features
df['month']=pd.to_datetime(df['OPENEDDATETIME']).dt.month.astype(object)
df['year']=pd.to_datetime(df['OPENEDDATETIME']).dt.year.astype(object)
df['day_of_week']=pd.to_datetime(df['OPENEDDATETIME']).dt.dayofweek.astype(object)
#output=r"C:\Users\Rayan\OneDrive\Documents\GitHub\311_services_city_SATX\311_services_city_SATX\src\data\SanAntonioCity311\311_data_lat_lon.csv"
#df.to_csv(output)

# %%
#Mapping
selected_df=df[[ 'lon','lat','Category','time_interval']]
WGS84 = CRS('epsg:4326')
crs=CRS('EPSG:3857')
geometry = [Point(xy) for xy in zip(selected_df.lon, selected_df.lat)]
geometry[:3]

# %%
# Map plotting
# Step 1: Create a GeoDataFrame

gdf = gpd.GeoDataFrame(selected_df, geometry=gpd.points_from_xy(selected_df.lon, selected_df.lat),crs=WGS84)
gdf.head()
print(gdf.crs)

gdf1=gdf.to_crs(crs)
print(gdf1.crs)


# %%

# Set the basemap CRS to WGS84
ax=gdf.plot(column='Category', categorical=True, legend=True,figsize=(10, 10),marker= '.',markersize=5,alpha=0.7 )
cx.add_basemap(ax=ax,crs="epsg:4326")
plt.saveas('allcategories.jpg')



# %% [markdown]
# ### The problem is that GeoJSON files tend to be large and an alternative would be using TopoJSON.

# %%
cat_n=gdf1["Category"].unique().tolist()
print(cat_n)
color_map=dict(zip(cat_n,sns.color_palette('husl', n_colors=len(cat_n))))
gdf1= gdf1.assign(color=df['Category'].map(color_map).map(rgb2hex))



# %%
# resampling becuase the dataset is large
# Define sample size
sample_size = 10000
# Stratified sampling (ensures class proportion is preserved)
gdf1_sample = resample(gdf1, n_samples=sample_size,
                       replace=False, stratify=gdf1['Category'])
#
m=folium.Map(location=(29.4, -98.50),max_bounds=False,width='80%',height='70%')
#color_name=pd.DataFrame(cnames,index=[0]).T.reset_index().rename(columns={'index':'color'})['color'].values[:(len(cat_n))]
color_name=['red', 'orange', 'yellow', 'green', 'blue', 'brown', 'violet', 'black'][:(len(cat_n))]
for cat, color in zip(cat_n,color_name):
    print(cat)
    #color=gdf1_sample[gdf1_sample["Category"] == cat]['color'].unique().tolist()
    print(color)
    #color=ListedColormap([color])
    #print(color) cmap=str(color),
    gdf1_sample[gdf1_sample["Category"] == cat].explore(m=m,categorical=True,
            legend=False,name=cat,marker_type='circle',
            marker_kwds=dict(radius=2,fill=True),color=color,
            style_kwds=dict(fill=True))

##
folium.TileLayer("CartoDB positron", show=False).add_to(m)
folium.TileLayer("OpenStreetMap").add_to(m)
folium.LayerControl().add_to(m)
# and then we write the map to disk
m.save('src\\visualization\categoriesSA.html')

# then open it
#webbrowser.open(r'/content/my_map.html')
#m



# %% [markdown]
# ##Folium Mapping

# %%
XX=round(gdf['geometry'].to_crs(wgs84_crs).centroid.x.mean(),2)
YY=round(gdf['geometry'].to_crs(wgs84_crs).centroid.y.mean(),2)
location=[YY,XX]
print(location)

# %% [markdown]
# ##Heat map Folium

# %%
m=folium.Map([YY,XX])
HeatMap(data=df[['lat', 'lon']].dropna().values.tolist(),radius=15,overlay=False,min_opacity=0.1).add_to(folium.FeatureGroup(name='Heat Map').add_to(m))
#HeatMap(data=df[['lat', 'lon', 'time_interval']].dropna().values.tolist(),radius=20,overlay=False,min_opacity=0.1).add_to(folium.FeatureGroup(name='Heat Map').add_to(m))
folium.LayerControl().add_to(m)
m.save("src\\visualization\SAN_density.html")
#m



