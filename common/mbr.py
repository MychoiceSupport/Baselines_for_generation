
try:
    from common.spatial_func import distance, SPoint
except:
    from spatial_func import distance,SPoint
class MBR:
    """
    MBR creates the minimal bounding regions for users.
    """
    def __init__(self, min_lat, min_lng, max_lat, max_lng):
        self.min_lat = min_lat
        self.min_lng = min_lng
        self.max_lat = max_lat
        self.max_lng = max_lng


    def get_range(self):
        print("查看范围min_lat, min_lng, max_lat, max_lng:", self.min_lat, self.min_lng, self.max_lat, self.max_lng)

    def contains(self, lat, lng):
        # return self.min_lat <= lat <= self.max_lat and self.min_lng <= lng <= self.max_lng
        # remove = max.lat/max.lng, to be consist with grid index
        return self.min_lat <= lat < self.max_lat and self.min_lng <= lng < self.max_lng

    def center(self):
        return (self.min_lat + self.max_lat) / 2.0, (self.min_lng + self.max_lng) / 2.0

    def get_h(self):
        print(f"{self.min_lat}, {self.min_lng}, {self.max_lat}, {self.max_lng}")
        return distance(SPoint(self.min_lat, self.min_lng), SPoint(self.max_lat, self.min_lng))

    def get_w(self):
        print(f"{self.min_lat}, {self.min_lng}, {self.max_lat}, {self.max_lng}")
        return distance(SPoint(self.min_lat, self.min_lng), SPoint(self.min_lat, self.max_lng))

    def __str__(self):
        h = self.get_h()
        w = self.get_w()
        return '{}x{}m2'.format(h, w)

    def __eq__(self, other):
        return self.min_lat == other.min_lat and self.min_lng == other.min_lng \
               and self.max_lat == other.max_lat and self.max_lng == other.max_lng

    def to_wkt(self):
        # Here providing five points is for GIS visualization
        # sometimes wkt cannot draw a rectangle without the last point.
        # (the last point should be the same as the first one)
        return 'POLYGON (({} {}, {} {}, {} {}, {} {}, {} {}))'.format(self.min_lng, self.min_lat,
                                                                      self.min_lng, self.max_lat,
                                                                      self.max_lng, self.max_lat,
                                                                      self.max_lng, self.min_lat,
                                                                      self.min_lng, self.min_lat)

    @staticmethod
    # staticmethod means this function will not use self attribute.
    def cal_mbr(coords):
        """
        Find MBR from coordinates
        Args:
        -----
        coords:
            list of Point()
        Returns:
        -------
        MBR()
        """
        min_lat = float('inf')
        min_lng = float('inf')
        max_lat = float('-inf')
        max_lng = float('-inf')
        for coord in coords:
            if coord.lat > max_lat:
                max_lat = coord.lat
            if coord.lat < min_lat:
                min_lat = coord.lat
            if coord.lng > max_lng:
                max_lng = coord.lng
            if coord.lng < min_lng:
                min_lng = coord.lng
        return MBR(min_lat, min_lng, max_lat, max_lng)

    @staticmethod
    def load_mbr(file_path):
        with open(file_path, 'r') as f:
            f.readline()
            attrs = f.readline()[:-1].split(';')
            mbr = MBR(float(attrs[1]), float(attrs[2]), float(attrs[3]), float(attrs[4]))
        return mbr

    @staticmethod
    def store_mbr(mbr, file_path):
        with open(file_path, 'w') as f:
            f.write('name;min_lat;min_lng;max_lat;max_lng;wkt\n')
            f.write('{};{};{};{};{};{}\n'.format(0, mbr.min_lat, mbr.min_lng, mbr.max_lat, mbr.max_lng, mbr.to_wkt()))
