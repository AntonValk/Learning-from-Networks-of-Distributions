import geopandas as gpd
from geopandas import GeoSeries
from shapely.geometry import Polygon
import shapely.geometry as geoms
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
from preprocess import preprocess_data_1


train_X, test_X, train_y, test_y = preprocess_data_1()


nybb_path = gpd.datasets.get_path('nybb')
boros = gpd.read_file(nybb_path)
boros.set_index('BoroCode', inplace=True)
boros.sort_index(inplace=True)
# print(boros.crs)
boros = boros.to_crs(epsg=4326)
print(boros.head())
boros.plot()
plt.show()


geos = gpd.GeoSeries(train_X[['longitude', 'latitude']]\
            .apply(lambda x: geoms.Point((x.longitude, x.latitude)), axis=1), \
            crs={'init': 'epsg:4326'})
# geos = geos.to_crs(epsg=2263)

# geos = gpd.GeoDataFrame(geos, geometry=gpd.points_from_xy(geos.longitude, geos.latitude))

gdf = gpd.GeoDataFrame(train_X, geometry=geos)


print(geos)

geos.plot()
plt.show()

# db['geometry'] = geos
# db = gpd.GeoDataFrame.from_records(db)
# db.crs = geos.crs

# only keep nyc 
in_nyc = gpd.sjoin(gdf, boros, how="inner", op='intersects')
# in_nyc = boros.contains(geos)

df = pd.DataFrame(in_nyc)
df = df[['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'num_photos', 'num_features', 'num_description_words', 'created_month', 'created_day', 'BoroName']]

# print(list(in_nyc))
# print(in_nyc['BoroName'])

print(df['BoroName'].value_counts())