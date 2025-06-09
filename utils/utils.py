from math import sin, cos, sqrt, atan2, radians, asin
import numpy as np
import torch


def resample_trajectory(x, length=200):
    """
    Resamples a trajectory to a new length.

    Parameters:
        x (np.ndarray): original trajectory, shape (N, 2)
        length (int): length of resampled trajectory

    Returns:
        np.ndarray: resampled trajectory, shape (length, 2)
    """
    len_x = len(x)
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T
    resampled_trajectory = np.zeros((2, length))
    for i in range(2):
        resampled_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[i])
    return resampled_trajectory.T


def time_warping(x, length=200):
    """
    Resamples a trajectory to a new length.
    """
    len_x = len(x)
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T
    warped_trajectory = np.zeros((2, length))
    for i in range(2):
        warped_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[i])
    return warped_trajectory.T


def gather(consts: torch.Tensor, t: torch.Tensor):
    """
    Gather consts for $t$ and reshape to feature map shape
    :param consts: (N, 1, 1)
    :param t: (N, H, W)
    :return: (N, H, W)
    """
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)


def q_xt_x0(x0, t, alpha_bar):
    # get mean and var of xt given x0
    mean = gather(alpha_bar, t) ** 0.5 * x0
    var = 1 - gather(alpha_bar, t)
    # sample xt from q(xt | x0)
    eps = torch.randn_like(x0).to(x0.device)
    xt = mean + (var ** 0.5) * eps
    return xt, eps  # also returns noise


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a


def p_xt(xt, noise, t, next_t, beta, eta=0):
    at = compute_alpha(beta.to(xt.device), t.long().to(xt.device))
    at_next = compute_alpha(beta.to(xt.device), next_t.long().to(xt.device))
    x0_t = (xt - noise * (1 - at).sqrt()) / at.sqrt()
    c1 = (eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt())
    c2 = ((1 - at_next) - c1 ** 2).sqrt()
    eps = torch.randn(xt.shape, device=xt.device)
    xt_next = at_next.sqrt() * x0_t + c1 * eps + c2 * noise
    return xt_next


def divide_grids(boundary, grids_num):
    lati_min, lati_max = boundary['lati_min'], boundary['lati_max']
    long_min, long_max = boundary['long_min'], boundary['long_max']
    # Divide the latitude and longitude into grids_num intervals.
    lati_interval = (lati_max - lati_min) / grids_num
    long_interval = (long_max - long_min) / grids_num
    # Create arrays of latitude and longitude values.
    latgrids = np.arange(lati_min, lati_max, lati_interval)
    longrids = np.arange(long_min, long_max, long_interval)
    return latgrids, longrids


# calculte the distance between two points
def distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000

def geodesic(pos1, pos2):
    try:
        pos1 = torch.from_numpy(pos1)
        pos2 = torch.from_numpy(pos2)
    except:
        pass
    lat0 = torch.deg2rad(pos1[..., 0])
    lat1 = torch.deg2rad(pos2[..., 0])
    lng0 = torch.deg2rad(pos1[..., 1])
    lng1 = torch.deg2rad(pos2[..., 1])
    dLat = lat1 - lat0
    dLon = lng1 - lng0
    a = torch.sin(dLat / 2) * torch.sin(dLat / 2) + torch.sin(dLon / 2) * torch.sin(dLon / 2) * torch.cos(
        lat0) * torch.cos(lat1)
    distance = 2 * 6371473 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return distance

# def calculate_route_dis(gps_slice, time_interval = 15):
#     """
#     :param gps_slice: T * 2
#     :return:
#     """
#     print(gps_slice.shape)
#     distance_mid = geodesic(gps_slice[:,:-1,], gps_slice[:,1:,])
#     print("查看距离形状:",distance_mid.shape)
#     avg_dis = torch.mean(distance_mid, dim=-1) * 1e3
#     avg_time = (len(gps_slice) - 1) * time_interval
#     traval_dis = geodesic(gps_slice[:,0], gps_slice[:,-1])
#     print("查看travel_dis的形状:", traval_dis.shape)
#     avg_speed = torch.sum(distance_mid, dim=-1) / ((len(gps_slice) - 1) * time_interval) * 1e3
#     print("查看speed形状:", avg_speed.shape)
#     num_point = len(gps_slice)
#     return avg_dis, avg_time, traval_dis, avg_speed, num_point

def calculate_route_dis(gps_slice, time_interval = 15, dataset='xian'):
    """
    :param gps_slice: T * 2
    :return:
    """
    # print(gps_slice.shape)
    lens = gps_slice.shape[0]
    if dataset != 'Porto':
        gps_slice = gps_slice.reshape(-1, lens, 2)
    try:
        distance_mid = geodesic(gps_slice[:,:-1,], gps_slice[:,1:,])
    except:
        distance_mid = geodesic(gps_slice[:-1,:], gps_slice[1:,:])
    # print("查看距离形状:",distance_mid.shape)
    avg_dis = torch.Tensor(torch.mean(distance_mid, dim=-1))
    avg_time = np.array([(gps_slice.shape[1] - 1) * time_interval])
    travel_dis = torch.sum(distance_mid, dim=-1)
    avg_speed = travel_dis / ((gps_slice.shape[1] - 1) * time_interval)
    # print("查看speed形状:", avg_dis.shape, traval_dis.shape, avg_speed.shape)
    if avg_dis.shape[0] == 0:
        print(avg_dis)
        print(gps_slice)
        raise Exception("数据有错")
    num_point = np.array([gps_slice.shape[1]] * gps_slice.shape[0])
    return avg_dis.reshape(-1,1), avg_time.reshape(-1,1), travel_dis.reshape(-1,1), avg_speed.reshape(-1,1), num_point.reshape(-1,1)
