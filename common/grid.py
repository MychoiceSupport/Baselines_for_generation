import numpy as np
from .mbr import MBR
from .spatial_func import LAT_PER_METER, LNG_PER_METER


class Grid:
    """
    index order
    30 31 32 33 34...
    20 21 22 23 24...
    10 11 12 13 14...
    00 01 02 03 04...
    """

    def __init__(self, mbr, row_num, col_num):
        self.mbr = mbr
        self.row_num = row_num
        self.col_num = col_num
        self.lat_interval = (mbr.max_lat - mbr.min_lat) / float(row_num)
        self.lng_interval = (mbr.max_lng - mbr.min_lng) / float(col_num)

    def get_row_idx(self, lat):
        row_idx = (lat - self.mbr.min_lat) // self.lat_interval
        # for data in row_idx:
        #     assert data >= 0
        # if row_idx.any() >= self.row_num or row_idx.any() < 0:
        #     raise IndexError("lat is out of mbr")
        ###这个部分需要能够处理row或者col的格式
        return row_idx

    def get_col_idx(self, lng):
        col_idx = (lng - self.mbr.min_lng) // self.lng_interval
        # for data in col_idx:
        #     assert data >= 0
        # if col_idx.any() >= self.col_num or col_idx.any() < 0:
        #     raise IndexError("lng is out of mbr")
        return col_idx

    def get_1d_idx(self, gpses):
        row_idx = self.get_row_idx(gpses[...,0])
        col_idx = self.get_col_idx(gpses[...,1])
        return row_idx * self.col_num + col_idx

    def safe_matrix_to_idx(self, lat, lng):
        try:
            return self.get_matrix_idx(lat, lng)
        except IndexError:
            return np.nan, np.nan

    def get_idx(self, lat, lng):
        return self.get_row_idx(lat), self.get_col_idx(lng)

    def get_matrix_idx(self, lat, lng):
        return self.row_num - 1 - self.get_row_idx(lat), self.get_col_idx(lng)

    def get_min_lng(self, col_idx):
        return self.mbr.min_lng + col_idx * self.lng_interval

    def get_max_lng(self, col_idx):
        return self.mbr.min_lng + (col_idx + 1) * self.lng_interval

    def get_min_lat(self, row_idx):
        return self.mbr.min_lat + row_idx * self.lat_interval

    def get_max_lat(self, row_idx):
        return self.mbr.min_lat + (row_idx + 1) * self.lat_interval

    # def get_mid(self, row_idx, col_idx):
    #     assert row_idx.shape == col_idx.shape
    #     min_lat = self.get_min_lat(row_idx)
    #     max_lat = self.get_max_lat(row_idx)
    #     min_lng = self.get_min_lng(col_idx)
    #     max_lng = self.get_max_lng(col_idx)
    #     return min_lat, min_lng, max_lat, max_lng

    def get_mbr_by_idx(self, row_idx, col_idx):
        min_lat = self.get_min_lat(row_idx)
        max_lat = self.get_max_lat(row_idx)
        min_lng = self.get_min_lng(col_idx)
        max_lng = self.get_max_lng(col_idx)
        return MBR(min_lat, min_lng, max_lat, max_lng)

    def get_mbr_by_matrix_idx(self, mat_row_idx, mat_col_idx):
        row_idx = self.row_num - 1 - mat_row_idx
        min_lat = self.get_min_lat(row_idx)
        max_lat = self.get_max_lat(row_idx)
        min_lng = self.get_min_lng(mat_col_idx)
        max_lng = self.get_max_lng(mat_col_idx)
        return MBR(min_lat, min_lng, max_lat, max_lng)


    def range_query(self, query_mbr, type):
        target_idx = []
        # squeeze the mbr a little, since the top and right boundary are belong to the other grid
        min_lat = max(query_mbr.min_lat, self.mbr.min_lat)
        min_lng = max(query_mbr.min_lng, self.mbr.min_lng)
        max_lat = min(query_mbr.max_lat, self.mbr.max_lat)
        max_lng = min(query_mbr.max_lng, self.mbr.max_lng)
        if type == 'matrix':
            max_row_idx, min_col_idx = self.get_matrix_idx(min_lat, min_lng)
            min_row_idx, max_col_idx = self.get_matrix_idx(max_lat, max_lng)
        elif type == 'cartesian':
            min_row_idx, min_col_idx = self.get_idx(min_lat, min_lng)
            max_row_idx, max_col_idx = self.get_idx(max_lat, max_lng)
        else:
            raise Exception('unrecognized index type')
        for r_idx in range(min_row_idx, max_row_idx + 1):
            for c_idx in range(min_col_idx, max_col_idx + 1):
                target_idx.append((r_idx, c_idx))
        return target_idx

    def get_coords_idx(self, coords, per_lat, per_lng):
        # print("查看coords的形状:", coords.shape)
        row_idx = (coords[..., 0] - self.mbr.min_lat) // per_lat
        col_idx = (coords[..., 1] - self.mbr.min_lng) // per_lng
        return np.concatenate([np.array(row_idx).reshape(-1,1), np.array(col_idx).reshape(-1,1)], axis=-1)


def create_grid(min_lat, min_lng, max_lat, max_lng, km_per_cell_lat, km_per_cell_lng):
    """
    Given region and unit of each cell, return a Grid class.
    Update original function since it's difficult to know the length of lat and lng.
    """
    print(f"查看:{min_lat},{min_lng},{max_lat},{max_lng}")
    mbr = MBR(min_lat, min_lng, max_lat, max_lng)
    km_lat = mbr.get_h()
    km_lng = mbr.get_w()
    print("km_lat与km_lng的内容:",km_lat,km_lng)
    nb_rows = int(km_lat / km_per_cell_lat)
    nb_cols = int(km_lng / km_per_cell_lng)
    print("查看Grid的num", nb_rows,nb_cols)
    return Grid(mbr, nb_rows, nb_cols)

def create_grid_num(min_lat, min_lng, max_lat, max_lng, nb_rows, nb_cols):
    mbr = MBR(min_lat, min_lng, max_lat, max_lng)
    return Grid(mbr, nb_rows, nb_cols)
