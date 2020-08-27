import geopandas as gpd
import networkx as nx
from geopandas import GeoSeries
from shapely.geometry import Point, Polygon
import shapely.geometry as geoms
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
from preprocess import preprocess_data_geog
from sklearn.model_selection import train_test_split
from shapely.ops import nearest_points
# from shapely.ops import voronoi_diagram
# from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area
from geovoronoi import points_to_coords
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import shapely.geometry
import shapely.ops

df = pd.read_csv('neighbourhood_data.csv')
counts = df['neighbourhood'].value_counts()
count_list = counts[counts > 50].index.tolist()
df = df[df['neighbourhood'].isin(count_list)]


districts = gpd.read_file('nyc_neighborhoods.geojson')
districts.sort_index(inplace=True)
districts = districts.to_crs(epsg=4326)
districts = districts.loc[:, ~districts.columns.str.contains('^Unnamed')]


coords = points_to_coords(districts.geometry)
vor = Voronoi(coords)
voronoi_plot_2d(vor)

plt.show()

lines = [shapely.geometry.LineString(vor.vertices[line])
    for line in vor.ridge_vertices
    if -1 not in line]

districts = districts[districts['name'].isin(df['neighbourhood'].unique())]

districts['voronoi_polygon'] = None
for index, row in districts.iterrows():
    for poly in shapely.ops.polygonize(lines):
        if(poly.intersects(row.geometry)):
            districts.at[index, 'voronoi_polygon'] = poly

districts = districts.dropna(axis = 0, how ='any') 

df = df[df['neighbourhood'].isin(districts['name'].unique())]
# new_index = np.arange(districts.shape[0])
# print(new_index)
# print(districts.count)
# id_num = pd.Series(new_index)
# print(districts['name'].nunique())
# districts.insert(loc=0, column='ID', value=id_num)
# districts['id'] = id
# districts = districts.set_index('ID')
districts.reset_index(drop=True, inplace=True)
# print(districts.head())
# print(districts)


polygons = gpd.GeoSeries(districts['voronoi_polygon'])
# districts['voronoi_polygon'] = polygons
# print(type(polygons))
# print(type(districts['voronoi_polygon']))
gdf = gpd.GeoDataFrame(districts, geometry=polygons)
# print(gdf.head)

# exit()
districts['neighbours'] = None


for index, row in gdf.iterrows():
    neighbours = gdf[gdf.geometry.touches(districts.at[index,'geometry'])].index
    neighbours = list(neighbours)
    # neighbours = neighbours[neighbours < 150]
    districts.at[index, "neighbours"] = neighbours

districts = districts[districts.astype(str)['neighbours'] != '[]']

# print(list(districts))
# exit()

# for index, row in districts.iterrows():
#     nb = districts.at[index,'neighbours']
#     for n in nb:
#         if n not in list(districts.Index):
#             nb.remove(n)


# exit()


# districts = districts.astype({'neighbours': 'int64'})
# districts = districts[districts['boro_cd'] < 150]

# districts = districts.drop(['voronoi_polygon'])
neighbourhoods = districts.iloc[:, -1]

# print(list(districts))
# print(districts)
# exit()
neighbourhoods = list(neighbourhoods)
##########################################
# print(neighbourhoods)
# # neighbourhoods = neighbourhoods[neighbourhoods.astype(bool)]
# print(len(neighbourhoods))

# # for index, row in districts.iterrows():
# #     if(len(row['neighbours']) == 0):
# #         districts.drop(row)

# neighbourhoods = [x for x in neighbourhoods if x != []]
# print(neighbourhoods)
# print(len(neighbourhoods))
###########################################
dicts = {}
# districts = districts.sort_values(by=['boro_cd'])

# print(districts["neighbours"])
# exit()

keys = list(range(len(neighbourhoods)))
values = neighbourhoods

assert (len(keys) == len(values))

for i,j in zip(keys, values):
        dicts[i] = j
# print(dicts)

g = nx.DiGraph(dicts)
g.remove_nodes_from(list(nx.isolates(g)))
outdeg = g.out_degree()
# print(outdeg)
to_remove = []
for elem in outdeg:
    if elem[1] == 0:
        to_remove.append(elem[0])
g.remove_nodes_from(to_remove)
outdeg = g.out_degree()
# print(len(dicts))

# nx.draw(g, with_labels = True)
# plt.show()

g = g.to_undirected()

assert(nx.number_connected_components(g) == 1)
adj = nx.adjacency_matrix(g).todense()
adj = adj + np.eye(adj.shape[0])
# adj = 0.5 * (adj + adj.T)
# print(adj)
# print(adj.shape)
# print(adj)
districts.to_csv('districts_data_cleaned.csv')
np.savetxt('adj_nbhd.txt', adj, delimiter=',')