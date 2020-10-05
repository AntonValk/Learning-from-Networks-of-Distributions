import geopandas as gpd
import networkx as nx
from geopandas import GeoSeries
from shapely.geometry import Polygon
from shapely.geometry import Point
import shapely.geometry as geoms
from shapely.ops import nearest_points
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
from preprocess import preprocess_data_geog
from sklearn.model_selection import train_test_split

df = pd.read_csv('neighbourhood_data.csv')
print(df['neighbourhood'].value_counts().describe())
counts = df['neighbourhood'].value_counts()
count_list = counts[counts > 50].index.tolist()
df = df[df['neighbourhood'].isin(count_list)]
print(df['neighbourhood'].value_counts().describe())


df = preprocess_data_geog()
geos = gpd.GeoSeries(df[['longitude', 'latitude']]\
            .apply(lambda x: geoms.Point((x.longitude, x.latitude)), axis=1), \
            crs={'init': 'epsg:4326'})
gdf = gpd.GeoDataFrame(df, geometry=geos)

districts = gpd.read_file('nyc_neighborhoods.geojson')
districts.sort_index(inplace=True)
districts = districts.to_crs(epsg=4326)
districts = districts.astype({'name': 'str'})


nybb_path = gpd.datasets.get_path('nybb')
boros = gpd.read_file(nybb_path)
boros.set_index('BoroCode', inplace=True)
boros.sort_index(inplace=True)
boros = boros.to_crs(epsg=4326)

gdf = gpd.sjoin(gdf, boros, how="inner", op='intersects')

gdf['neighbourhood'] = None  # add column

pts3 = districts.geometry.unary_union
def near(point, pts=pts3):
     nearest = districts.geometry == nearest_points(point, pts)[1]
     return str(districts.name[nearest].values[0])

gdf['neighbourhood'] = gdf.apply(lambda row: near(row.geometry), axis=1)


nyc = pd.DataFrame(gdf)

num_feats = ["bathrooms", "bedrooms", "interest_level",
             "num_photos", "num_features", "num_description_words",
             "created_month", "created_day", "neighbourhood"]

X = nyc[num_feats]
y = nyc["price"]


num_feats_2 = ["bathrooms", "bedrooms", "interest_level",
             "num_photos", "num_features", "num_description_words",
             "created_month", "created_day", "neighbourhood", "price"]

nyc = nyc[num_feats_2]

nyc.to_csv('neighbourhood_data.csv')

print(nyc.head())

print(X['neighbourhood'].value_counts())
print(X['neighbourhood'].value_counts().describe())


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)

print(X_train.shape, y_train.shape)