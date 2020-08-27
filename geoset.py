import geopandas as gpd
import networkx as nx
from geopandas import GeoSeries
from shapely.geometry import Polygon
import shapely.geometry as geoms
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
from preprocess import preprocess_data_geog
from sklearn.model_selection import train_test_split



# df = preprocess_data_geog()


nybb_path = gpd.datasets.get_path('nybb')
boros = gpd.read_file(nybb_path)
boros.set_index('BoroCode', inplace=True)
boros.sort_index(inplace=True)
# print(boros.crs)
boros = boros.to_crs(epsg=4326)
# print(boros.head())
# boros.plot()
# plt.show()


districts = gpd.read_file('nyc_districts.geojson')
districts.sort_index(inplace=True)
# print(boros.crs)
districts = districts.to_crs(epsg=4326)
# print(districts.head())
# districts.plot()
# plt.show()



# neighbors = np.array(districts[districts.geometry.touches(districts['geometry'])].boro_cd)
# #overlapping neighbors use if discrepances found with touches
# overlap = np.array(districts[districts.geometry.overlaps(districts['geometry'])].boro_cd)

# neighbors = np.union1d(neighbors, overlap)

districts["neighbours"] = None  # add column
# districts = districts[districts['boro_cd'] < 200]


for index, row in districts.iterrows():  
    neighbours = districts[districts.geometry.touches(row['geometry'])].boro_cd.to_numpy() 
    neighbours = neighbours.astype(np.int64)
    neighbours = neighbours[neighbours < 150]
    districts.at[index, "neighbours"] = neighbours

districts = districts.astype({'boro_cd': 'int64'})
districts = districts[districts['boro_cd'] < 150]

 
x = np.array(districts['neighbours'])




dicts = {}
districts = districts.sort_values(by=['boro_cd'])

print(districts[["boro_cd", "neighbours"]])

keys = districts['boro_cd'].to_numpy()
values = districts['neighbours'].to_numpy()


for i,j in zip(keys, values):
        dicts[i] = j
# print(dicts)

g = nx.Graph(dicts)
# nx.draw(g, with_labels = True)
# plt.show()


adj = nx.adjacency_matrix(g)
adj = adj.todense() + np.eye(districts['boro_cd'].nunique())
np.savetxt('adj.txt', adj, delimiter=',')


exit()


geos = gpd.GeoSeries(df[['longitude', 'latitude']]\
            .apply(lambda x: geoms.Point((x.longitude, x.latitude)), axis=1), \
            crs={'init': 'epsg:4326'})
# geos = geos.to_crs(epsg=2263)

# geos = gpd.GeoDataFrame(geos, geometry=gpd.points_from_xy(geos.longitude, geos.latitude))

gdf = gpd.GeoDataFrame(df, geometry=geos)

# print(geos)
# geos.plot()
# plt.show()


# only keep nyc 
in_nyc_2 = gpd.sjoin(gdf, boros, how="inner", op='intersects')
# nyc_2 = pd.DataFrame(in_nyc_2)

in_nyc_2 = in_nyc_2.drop(['index_right'], axis=1)

in_nyc = gpd.sjoin(in_nyc_2, districts, how="inner", op='intersects')
# in_nyc = boros.contains(geos)

nyc = pd.DataFrame(in_nyc)
# in_nyc.plot()
# plt.show()

######################################################### uncomment to plot
# base = districts.plot(color='white', edgecolor='black')

# in_nyc.plot(ax=base, marker='+', color='red', markersize=2)
# plt.title('GPS Map of NYC Rental Listings')
# plt.show()
######################################################### uncomment to plot


num_feats = ["bathrooms", "bedrooms", "interest_level",
             "num_photos", "num_features", "num_description_words",
             "created_month", "created_day", "boro_cd"]

# nyc = nyc[nyc['BoroName'] == 'Manhattan']
# nyc = nyc[nyc['boro_cd'] < 200]
X = nyc[num_feats]
# print(X['BoroName'].value_counts())
# # countries = world[world['continent'] == "South America"]
# X = X[X['BoroName'] == 'Manhattan']
# print(X['BoroName'].value_counts())
y = nyc["price"]


# print(list(in_nyc))
# print(in_nyc['BoroName'])

# print(X['BoroName'].value_counts())

num_feats_2 = ["bathrooms", "bedrooms", "interest_level",
             "num_photos", "num_features", "num_description_words",
             "created_month", "created_day", "boro_cd", "price"]

nyc = nyc[num_feats_2]

nyc.to_csv('districts_data.csv')

print(nyc.head())

print(X['boro_cd'].value_counts())



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)

print(X_train.shape, y_train.shape)