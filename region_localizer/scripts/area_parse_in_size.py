"""Выгрузить OSM изображение заданного размера в заданной координате."""


import osmnx as ox
import geopandas as gpd
import pymap3d as pm
import sys
import matplotlib.pyplot as plt
from pathlib import Path


def geodetic2enu(origin_lat, origin_lon, lat, lon):
    alt0 = 0  # высота цели над геодезическим эллипсоидом (метры)
    alt = 0
    # Север, Восток, Вниз (NED), специально используется в аэрокосмической отрасли https://en.wikipedia.org/wiki/Axes_conventions#Ground_reference_frames:_ENU_and_NED
    # geodetic2ned(lat, lon, h, lat0, lon0, h0, ell=None, deg=True)
    x, y, z = pm.geodetic2enu(lat, lon, alt, origin_lat, origin_lon, alt0)
    # print('z', z)# перепад высоты по элипсу Земли
    return x, y

def enu2geodetic(target_X, target_Y, origin_lat, origin_lon):
    alt0 = 0  # высота цели над геодезическим эллипсоидом (метры)
    z = 0
    # Север, Восток, Вниз (NED), специально используется в аэрокосмической отрасли https://en.wikipedia.org/wiki/Axes_conventions#Ground_reference_frames:_ENU_and_NED
    # geodetic2ned(lat, lon, h, lat0, lon0, h0, ell=None, deg=True)
    lat, lon, alt = pm.enu2geodetic(
        target_X, target_Y, z, origin_lat, origin_lon, alt0)
    return lat, lon


def get_area_cord(center_point, hight, width):
    origin_lat = center_point[0]
    origin_lon = center_point[1]
    
    lat_1, lon_1 = enu2geodetic(
        hight / 2, width / 2, origin_lat, origin_lon)
    lat_2, lon_2 = enu2geodetic(
        -hight / 2, -width / 2, origin_lat, origin_lon)
    
    return lat_1, lat_2, lon_1, lon_2


center_point = [49.992167, 36.231202]  # Геграфические координта цента искомой области
hight = 20000  # Высота выгружаемой области в метрах
width = 20000  # Ширина выгружаемой области в метрах

lat_1, lat_2, lon_1, lon_2 = get_area_cord(center_point, hight, width)
print(lat_1, lat_2, lon_1, lon_2)

# Тэги для выгружаемых примитивов
tags_all = {'natural': 'wood', 'landuse': 'forest', 'highway': True}
map_landmark_all = ox.features_from_bbox(
    bbox=(lat_1, lat_2, lon_1, lon_2), tags=tags_all)

map_landmark_all = map_landmark_all.loc[
    :, map_landmark_all.columns.str.contains('addr:|geometry')]

name_vector_map = 'map_' + \
    str(round(center_point[0])) + '_' + \
    str(round(center_point[1])) + '.geojson'
save_pth = Path('data/geo_osm')
map_landmark_all.to_file(save_pth / name_vector_map, driver='GeoJSON')

df = gpd.read_file(save_pth / name_vector_map)

ax = df.plot(color='k', linewidth=0.1, markersize=0.1)
ax.set_xlim(lon_1, lon_2)
ax.set_ylim(lat_1, lat_2)


plt.xticks([])
plt.yticks([])
ax.axis('off')

name_vector_map = 'map_' + \
    str(round(center_point[0])) + '_' + \
    str(round(center_point[1])) + '.png'
plt.savefig(save_pth / name_vector_map, bbox_inches='tight', pad_inches=0, dpi=300)
plt.show()
sys.exit()
