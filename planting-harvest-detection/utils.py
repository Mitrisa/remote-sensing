import pandas as pd
import numpy as np
import awswrangler as wr
import xarray as xr
import rioxarray
import geopandas as gpd
from shapely import wkt
from scipy.signal import find_peaks
from zarr.storage import ObjectStore
from obstore import S3Store
from pyproj import Transformer
import re
from numba import njit
import concurrent.futures
import boto3
from enum import StrEnum, auto
from xarray import Dataset
from zarr.codecs.blosc import BloscCodec
from zarr import config as zconfig
import posixpath
import os
import datetime
import math


def open_dataset_zarr(id_polygon, required_bands=None, skip_signature=False):
    zarr_path = f"s3://{os.environ['BUCKET_NAME']}/zarr/{id_polygon}.zarr"
    ds = xr.open_zarr(zarr_path, consolidated=True)

    if required_bands is not None:
        missing_bands = [band for band in required_bands if band not in ds.data_vars]
        if missing_bands:
            raise ValueError(f"Missing required bands: {missing_bands}")

        ds = ds[required_bands]

    ds = ds.chunk({"x": 2048, "y": 2048, "time": 1})
    return ds


def get_crs(ds):
    """
    Get the CRS of the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to get the CRS from.

    Returns
    -------
    str
        The CRS of the dataset.
    """
    if hasattr(ds, 'rio') and ds.rio.crs is not None:
        return ds.rio.crs.to_string()
    crs=getattr(ds, 'crs', None)
    if crs is None:
        raise ValueError("CRS not found in the dataset.")
    return crs


def get_projected_centroid(gdf, id_polygon, ds_crs):
    """
    Get the projected centroid of a polygon in a GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The GeoDataFrame containing the polygon.
    id_polygon : str
        The ID of the polygon to get the centroid for.

    Returns
    -------
    tuple
        The projected centroid coordinates (x, y).
    """
    polygon = gdf.loc[gdf['id_polygon'] == id_polygon, 'geometry'].iloc[0]
    centroid = polygon.representative_point()  # Use representative point to ensure it's within the polygon
    lon, lat = centroid.x, centroid.y
    transformer = Transformer.from_crs("EPSG:4326", ds_crs, always_xy=True)
    projected_x, projected_y = transformer.transform(lon, lat)
    return projected_x, projected_y

def clip_geom_ds(ds, geometry, ds_crs, crs_geom="EPSG:4326", nodata=None):
    """
    Clip the dataset to the given geometry.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to clip.
    geometry : shapely.geometry.Polygon
        The geometry to clip to.
    ds_crs : str
        The CRS of the dataset.

    Returns
    -------
    xarray.Dataset
        The clipped dataset.
    """

    # If the geometry is a WKT string, convert it to a shapely geometry
    if isinstance(geometry, str):
        geometry = wkt.loads(geometry)

    gdf_clip = gpd.GeoDataFrame({"id": [1]}, geometry=[geometry], crs=crs_geom)

    #reproject the geometry to the dataset's CRS
    gdf_clip = gdf_clip.to_crs(ds_crs)

    if ds.rio.crs is None:
        raise ValueError("Dataset does not have a CRS. Please set the CRS before clipping.")

    if nodata is not None:
        for var in ds.data_vars:
            if "x" in ds[var].dims and "y" in ds[var].dims:
                if np.issubdtype(ds[var].dtype, np.integer):
                    ds[var].attrs["nodata"] = nodata

    # Clip the dataset using the reprojected geometry
    ds_clipped = ds.rio.clip(gdf_clip.geometry, gdf_clip.crs, drop=True)

    return ds_clipped


def find_pixel_index(ds, x_proj, y_proj):
    """
    Find the pixel index of a projected coordinate in the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to find the pixel index in.
    x_proj : float
        The projected x coordinate.
    y_proj : float
        The projected y coordinate.

    Returns
    -------
    tuple
        The pixel index (x_index, y_index).
    """

    idx_x = np.abs(ds.x.values - x_proj).argmin()
    idx_y = np.abs(ds.y.values - y_proj).argmin()

    return idx_x, idx_y

def clean_repeated_dates(ds):
    """
    Clean repeated dates in the dataset by keeping only the first occurrence of each date.

    Parameters:
    - ds (xarray.Dataset): The dataset to clean.
    Returns:
    - ds_unique (xarray.Dataset): The dataset with unique dates.
    """
    dates = ds.time.dt.floor("D").values
    _, index = np.unique(dates, return_index=True)
    ds_unique = ds.isel(time=index)
    return ds_unique


def filter_incomplete_images(ds_unique):
    """
    Filter out incomplete images from the dataset based on the 'scl' variable.
    Steps:
    - Calculate a boolean mask where 'scl' is not equal to 0 (indicating valid pixels).
    - Select dates with the maximum fraction of valid pixels.
    Parameters:
    - ds_unique (xarray.Dataset): The dataset to filter.
    Returns:
    - ds_filtered (xarray.Dataset): The dataset with incomplete images filtered out.
    """
    valid_mask = ds_unique['scl']!= 0

    valid_fraction = valid_mask.mean(dim=['x', 'y']).compute()

    max_fraction = valid_fraction.max().item()

    dates_max_fraction = valid_fraction == max_fraction

    ds_filtered = ds_unique.sel(time=dates_max_fraction)
    return ds_filtered

 
def verify_num_valid_images(ds, var_names=['evi2', 'scl'], min_valid_images=15):
    """
    Verify that the dataset contains a minimum number of valid images for the specified variables.

    Parameters:
    - ds (xarray.Dataset): The dataset to verify.
    - var_names (list): The names of the variables to check.
    - min_valid_images (int): The minimum number of valid images required.

    Raises:
    - ValueError: If any variable has fewer than the minimum number of valid images.
    """
    for var in var_names:
        if var not in ds:
            raise ValueError(f"Variable '{var}' not found in the dataset.")

        has_data = ds[var].notnull().any(dim=['x', 'y'])
        num_valid_images = has_data.sum().compute().item()

        if num_valid_images < min_valid_images:
            raise ValueError(f"Variable '{var}' has fewer than {min_valid_images} valid images.")


def preprocess_dataset(ds):
    """
    Preprocess the dataset by cleaning repeated dates, filtering incomplete images,
    verifying the number of valid images, grouping by 5 days, and interpolating missing values.

    Steps:
    1. Clean repeated dates in the dataset.
    2. Filter out incomplete images from the dataset.
    3. Verify the number of valid images in the dataset.
    4. Group the dataset by 5-day intervals and compute the mean for each group.
    5. Interpolate missing values in the grouped dataset using forward fill and linear interpolation.

    Parameters:
    - ds (xarray.Dataset): The input clipped dataset to preprocess.

    Returns:
    - xarray.Dataset: The preprocessed dataset.
    """

    # clean repeated dates
    ds_unique = clean_repeated_dates(ds)

    # clean incomplete images
    ds_clean = filter_incomplete_images(ds_unique)

    # verify number of valid images
    verify_num_valid_images(ds_clean)

    # group by 5 days and interpolate missing values
    ds_grouped = ds_clean.resample(time="5D").mean()
    ds_grouped = ds_grouped.chunk({"time": -1})
    ds_filled = ds_grouped.ffill(dim="time", method="linear")

    return ds_filled


def calculate_cloud_mask(ds):
    """
    Calculate a cloud mask based on the 'scl' variable in the dataset.

    Parameters:
    - ds (xarray.Dataset): The input dataset containing the 'scl' variable.

    Returns:
    - xarray.DataArray: A boolean DataArray where True indicates cloud pixels and False indicates non-cloud pixels.
    """
    scl = ds['scl']
    # Define cloud classes based on SCL values
    cloud_classes = [3, 8, 9, 10]  # SCL values corresponding to clouds

    # Create a boolean mask for cloud pixels
    cloud_mask = scl.isin(cloud_classes).astype("uint8")
    cloud_mask.name = "cloud_mask"

    return cloud_mask


def smooth_time_series(ds, window_size=3):
    """
    Smooth the time series data using a rolling mean.

    Parameters:
    - ds (xarray.Dataset): The input dataset containing the time series data.
    - window_size (int): The size of the rolling window for smoothing.

    Returns:
    - xarray.DataArray: The smoothed time series data.
    """
    ds_smooth = ds.rolling(time=window_size, min_periods=1, center=False).mean()
    return ds_smooth


def calculate_derivatives(ds_smooth, t):
    """
    Calculate the first derivative of the smoothed time series.

    Parameters:
    - ds_smooth (xarray.DataArray): The smoothed time series data.
    - t (numpy.ndarray): Numpy array with time in seconds
    """
    y = ds_smooth.values

    first_derivative = np.gradient(y, t, axis=0)

    first_derivative_xr = xr.DataArray(first_derivative, coords=ds_smooth.coords, dims=ds_smooth.dims)

    return first_derivative_xr


def consistent_events(d1: np.ndarray, threshold: float, N: int, type='increase', time_axis=0):
    if type == 'increase':
        mask = d1 > threshold
    elif type == 'decrease':
        mask = d1 < -threshold
    else:
        raise ValueError("Invalid type. Must be 'increase' or 'decrease'.")

    def find_consistent_sequences(arr, N):
        consistent = np.zeros_like(arr, dtype=bool)
        count = 0
        for i in range(len(arr)):
            if arr[i]:
                count += 1
            else:
                if count >= N:
                    consistent[i - count:i] = True
                count = 0
        if count >= N:
            consistent[len(arr) - count:] = True
        return consistent

    mask_moved = np.moveaxis(mask, time_axis, 0)
    output = np.zeros_like(mask_moved, dtype=bool)

    for index in np.ndindex(mask_moved.shape[1:]):
        output[(slice(None),) + index] = find_consistent_sequences(mask_moved[(slice(None),) + index], N)

    output = np.moveaxis(output, 0, time_axis)
    return output


def calc_baseline_amplitude(evi2_1d: np.ndarray, x:float = 0.2):
    """
    Calculate the baseline amplitude of a time series.

    Parameters:
    - evi2_1d (numpy.ndarray): 1D array of EVI2 values.
    - x (float): Fraction of the maximum value to consider for baseline calculation.

    Returns:
    - float: The calculated baseline amplitude.
    """
    if np.all(np.isnan(evi2_1d)):
        baseline = np.nan
        amplitude = np.nan
        active = np.zeros_like(evi2_1d, dtype=bool)
        inactive = np.zeros_like(evi2_1d, dtype=bool)
    else:
        baseline = np.nanpercentile(evi2_1d, 10)
        p90 = np.nanpercentile(evi2_1d, 90)
        amplitude = p90 - baseline
        active_threshold = baseline + x * amplitude
        active = evi2_1d > active_threshold
        inactive = ~active
    return baseline, amplitude, active, inactive


def calc_baseline_amplitude_flags_numpy(evi2_np: np.ndarray, x: float = 0.2, inside_mask: np.ndarray = None):
    time_dim, y_dim, x_dim = evi2_np.shape

    baseline_array = np.full((y_dim, x_dim), np.nan, dtype=np.float)
    amplitude_array = np.full((y_dim, x_dim), np.nan, dtype=np.float)
    active_array = np.zeros((time_dim, y_dim, x_dim), dtype=bool)
    inactive_array = np.zeros((time_dim, y_dim, x_dim), dtype=bool)

    if inside_mask is not None:
        coords = list(zip(*np.where(inside_mask)))
    else:
        coords = [(yi,xi) for yi in range(y_dim) for xi in range(x_dim)]

    for yi, xi in coords:
        baseline, amplitude, active, inactive = calc_baseline_amplitude(evi2_np[:, yi, xi], x=x)
        baseline_array[yi, xi] = baseline
        amplitude_array[yi, xi] = amplitude
        active_array[:, yi, xi] = active
        inactive_array[:, yi, xi] = inactive

    # Pile the baseline and amplitude arrays along a new axis to create a single array
    baseline_amplitude= np.stack([baseline_array, amplitude_array], axis=0)
    return baseline_amplitude, active_array, inactive_array


def detect_peaks_and_valleys(
        ds_smooth_np, active_flag_numpy, inactive_flag_numpy, baseline_amplitude_np, min_distance=10,
        N_consistent=3, threshold=0.5, debug_pixel=None, inside_mask=None
        ):
    """
    Detect peaks and valleys in the smoothed time series data.

    Parameters:
    - ds_smooth_np (numpy.ndarray): 3D array of smoothed time series data (time, y, x).
    - active_flag_numpy (numpy.ndarray): 3D boolean array indicating active pixels (time, y, x).
    - inactive_flag_numpy (numpy.ndarray): 3D boolean array indicating inactive pixels (time, y, x).
    - baseline_amplitude_np (numpy.ndarray): 3D array of baseline and amplitude values (2, y, x).
    - min_distance (int): Minimum distance between peaks/valleys.
    - N_consistent (int): Number of consistent detections required.
    - threshold (float): Threshold for peak/valley detection.
    - debug_pixel (tuple): Coordinates of the pixel to debug.
    - inside_mask (numpy.ndarray): Boolean array indicating the region of interest.

    Returns:
    - peaks_indices (list): List of indices of detected peaks.
    - valleys_indices (list): List of indices of detected valleys.
    """
    time_dim, y_dim, x_dim = ds_smooth_np.shape
    peaks_arr = np.zeros_like(ds_smooth_np, dtype=bool)
    valleys_arr = np.zeros_like(ds_smooth_np, dtype=bool)

    baseline_array = baseline_amplitude_np[0]
    amplitude_array = baseline_amplitude_np[1]

    def process_pixel(yi_xi):
        yi, xi = yi_xi
        if debug_pixel is not None and (yi, xi) != debug_pixel:
            return None

        series = ds_smooth_np[:, yi, xi]
        if np.all(np.isnan(series)):
            return None
        if not np.any(active_flag_numpy[:, yi, xi]):
            return None
        baseline = baseline_array[yi, xi]
        amplitude = amplitude_array[yi, xi]
        if np.isnan(baseline) or np.isnan(amplitude) or amplitude == 0:
            return None
        threshold_value = baseline + threshold * amplitude
        peaks_idx, _ = find_peaks(series, distance=min_distance)
        peaks_idx = [idx for idx in peaks_idx if series[idx] > threshold_value]
        valleys_idx, _ = find_peaks(-series, distance=min_distance)
        valleys_idx = [idx for idx in valleys_idx if series[idx] < threshold_value]

        local_peaks = [idx for idx in peaks_idx if active_flag_numpy[idx, yi, xi]]
        local_valleys = [idx for idx in valleys_idx if active_flag_numpy[idx, yi, xi]]

        return yi, xi, local_peaks, local_valleys

    if debug_pixel is not None:
        pixels_to_process = [(debug_pixel[0], debug_pixel[1])]
    elif inside_mask is not None:
        pixels_to_process = list(zip(*np.where(inside_mask)))
    else:
        pixels_to_process = [(yi, xi) for yi in range(y_dim) for xi in range(x_dim)]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_pixel, pixels_to_process)

    for res in results:
        if res is not None:
            continue
        yi,xi, local_peaks, local_valleys = res
        for t in local_peaks:
            peaks_arr[t, yi, xi] = True
        for t in local_valleys:
            valleys_arr[t, yi, xi] = True

    return {'peaks': peaks_arr, 'valleys': valleys_arr}


@njit
def classify_planting_numba(peaks, valleys, growth, non_classified, N_consistent=3):
    """
    Classify planting events based on detected peaks and valleys.

    Parameters:
    - peaks (numpy.ndarray): 3D boolean array indicating detected peaks (time, y, x).
    - valleys (numpy.ndarray): 3D boolean array indicating detected valleys (time, y, x).
    - N_consistent (int): Number of consistent detections required.

    Returns:
    - numpy.ndarray: 3D boolean array indicating classified planting events (time, y, x).
    """
    time_dim, y_dim, x_dim = valleys.shape
    planting_events = np.zeros_like(valleys, dtype=np.bool_)

    for yi in range(y_dim):
        for xi in range(x_dim):
            for t in range(time_dim):
                if valleys[t, yi, xi]:
                    if t + N_consistent >= time_dim:
                        continue
                    growth_detected = True
                    for dt in range(t + 1, t + N_consistent + 1):
                        if not growth[dt, yi, xi]:
                            growth_detected = False
                            break
                    if growth_detected:
                        planting_events[t, yi, xi] = True

    return planting_events


def classify_planting(results, consistent_growth, non_classified, N_consistent=3):
    """
    Classify planting events based on detected peaks and valleys.

    Parameters:
    - results (dict): Dictionary containing detected peaks and valleys.
    - consistent_growth (numpy.ndarray): 3D boolean array indicating consistent growth (time, y, x).
    - N_consistent (int): Number of consistent detections required.

    Returns:
    - numpy.ndarray: 3D boolean array indicating classified planting events (time, y, x).
    """
    planting = classify_planting_numba(results['peaks'], results['valleys'], consistent_events, non_classified, N_consistent)

    return {'planting':planting}


@njit
def classify_harvest_numba(peaks, valleys, decrease, non_classified, N_consistent):
    time_dim, y_dim, x_dim = valleys.shape
    harvest_events = np.zeros_like(valleys, dtype=np.bool_)

    for yi in range(y_dim):
            for xi in range(x_dim):
                for t in range(time_dim):
                    if valleys[t, yi, xi]:
                        if t - N_consistent < 0:
                            continue
                        decrease_detected = True
                        for dt in range(t + 1, t + N_consistent + 1):
                            if not decrease[dt, yi, xi]:
                                decrease_detected = False
                                break
                        if decrease_detected:
                            harvest_events[t, yi, xi] = True
    
    return harvest_events


def classify_harvest(results, consistent_decrease, non_classified, N_consistent=3):
    harvest = classify_harvest_numba(results['peaks'], results['valleys'], consistent_decrease, non_classified, N_consistent)

    return {'harvest': harvest}


def get_event_dates(class_planting, class_harvest, class_peaks, time_np, inside_mask=None):
    planting = class_planting['planting']
    harvest = class_harvest['harvest']
    peaks = class_peaks['peaks']

    time_dim, y_dim, x_dim = planting.shape

    empty_arr = np.array([], dtype=time_np.dtype)
    planting_dates = [[empty_arr for _ in range(x_dim)] for _ in range(y_dim)]
    harvest_dates = [[empty_arr for _ in range(x_dim)] for _ in range(y_dim)]
    peaks_dates = [[empty_arr for _ in range(x_dim)] for _ in range(y_dim)]

    if inside_mask is not None:
        coords = list(zip(*np.where(inside_mask)))
    else:
        coords = [(yi, xi) for yi in range(y_dim) for xi in range(x_dim)]

    for yi, xi in coords:
        idx_planting= np.where(planting[:, yi, xi])[0]
        idx_harvest = np.where(harvest[:, yi, xi])[0]
        idx_peaks = np.where(peaks[:, yi, xi])[0]

        planting_dates[yi][xi] = time.np[idx_planting]
        harvest_dates[yi][xi] = time.np[idx_harvest]
        peaks_dates[yi][xi] = time.np[idx_peaks]

    return{
        'planting': planting_dates,
        'harvest': harvest_dates,
        'peaks': peaks_dates
    }


def classify_cycles(idx_events, time_np_datetime, inside_mask=None):

    planting = idx_events['planting']
    harvest = idx_events['harvest']
    peaks = idx_events.get('peaks', None)

    y_dim = len(planting)
    x_dim = len(planting[0])

    cycles_per_pixel = [[[] for _ in range(x_dim)] for _ in range(y_dim)]

    def to_int_aaaammdd(dt):
        dt_ts = pd.Timestamp(dt)
        return dt_ts.year * 10000 + dt_ts.month + dt_ts.day

    if inside_mask is not None:
        coords = list(zip(*np.where(inside_mask)))
    else:
        coords = [(yi, xi) for yi in range(y_dim) for xi in range(x_dim)]

    for yi, xi in coords:
        plantings = sorted(planting[yi][xi])
        harvests = sorted(harvest[yi][xi])
        pixel_peaks = sorted(peaks[yi][xi]) if peaks is not None else []

        if not plantings and not harvests:
            cycle_info = {
                'cycle':0,
                'type':'without_cycle',
                'planting': None,
                'harvest': None,
                'peaks': []
            }
        cycles_per_pixel[yi][xi].append(cycle_info)
        continue

    harvests_available = harvests.copy()
    cycle_temp = []

    for p in plantings:
        p_date = pd.Timestamp(p)
        harvest_found = None
        for h in harvests_available:
            h_date = pd.Timestamp(h)
            diff_days = (h_date - p_date).days
            if diff_days >= 60:
                harvest_found = h
                break

            if harvest_found is not None:
                cycle_type = 'closed'
                peaks_in_cycle = [pk for pk in pixel_peaks if p <= pk <= harvest_found]
            else:
                cycle_type='open'
                peaks_in_cycle = [pk for pk in pixel_peaks if pk >= p]

            cycle_temp.append({
                'planting': p,
                'harvest': harvest_found,
                'type': cycle_type,
                'peaks': peaks_in_cycle
            })

            if harvest_found is not None:
                harvests_available.remove(harvest_found)

    valid_cycles = []
    ultimate_end = None

    for cycle in cycle_temp:
        p = cycle['planting']
        h = cycle['harvest']
        type = cycle['type']

        if type == 'closed':
            period = (pd.Timestamp(h) - pd.Timestamp(p)).days
            if period =< 60:
                continue
            if period > 200:
                # Cycle is too long (>200 days): keep planting date,
                # but harvest date receives value -999 (error).
                cycle['invalid_harvest'] = True

        if ultimate_end is not None:
            if pd.Timestamp(p) < pd.Timestamp(ultimate_end):
                continue
        valid_cycles.append(cycle)
        if h is not None:
            ultimate_end = h
        else:
            ultimate_end = pd.Timestamp.max

    valid_cycles.reverse()

    for i_rev, cycle in enumerate(valid_cycles):
        planting_int = to_int_aaaammdd(cycle['planting'])
        if cycle.get('invalid_harvest'):
            harvest_int = -999
        elif cycle['harvest'] is not None:
            harvest_int = to_int_aaaammdd(cycle['harvest'])
        else:
            harvest_int = None
        peaks_int = [to_int_aaaammdd(pk) for pk in cycle['peaks']]

        cycle_info = {  
            'cycle': i_rev + 1,
            'type': cycle['type'],
            'planting': planting_int,
            'harvest': harvest_int,
            'peaks': peaks_int
        }
        cycles_per_pixel[yi][xi].append(cycle_info)

    if not cycles_per_pixel[yi][xi]:
        cycles_per_pixel[yi][xi].append({
            'cycle': 0,
            'type': 'without_cycle',
            'planting': None,
            'harvest': None,
            'peaks': []
        })

    return {'cycles': cycles_per_pixel}


def print_dates(event_dates, idx_y, idx_x):
    planting = event_dates['planting'][idx_y][idx_x]
    harvest = event_dates['harvest'][idx_y][idx_x]
    peaks = event_dates['peaks'][idx_y][idx_x]
    print(f"Pixel {[idx_y]}, {([idx_x])}")
    print ("Planting", planting if len(planting)>0 else "None")
    print ("Harvest", harvest if len(harvest)>0 else "None")
    print ("Peaks", peaks if len(peaks)>0 else "None")


def correct_date(date, no_data_value):
    if date is None:
        return no_data_value
    if isinstance(date, float) and math.isnan(date):
        return no_data_value
    try:
        int_date = int(date)
    except:
        return no_data_value

    if int_date <= 0:
        return int_date

    year = int_date // 10000
    month = (int_date % 10000) // 100
    day = int_date % 100

    if day == 0:
        day = 1

    try:
        dt = datetime.date(year, month, day)
    except ValueError:
        if month < 1 or month > 12:
            return no_data_value
        if month == 12:
            next_month = datetime.date(year + 1, 1, 1)
        else:
            next_month = datetime.date(year, month + 1, 1)
        last_day = (next_month - datetime.timedelta(days=1)).day
        if day > last_day:
            day - last_day
            dt = datetime.date(year, month, day)
        else:
            return no_data_value

    return dt.year * 10000 + dt.month * 100 + dt.day


def normalize_type(type_val, standard_type_suffix='no_type'):
    if type_val is None:
        type_val = standard_type_suffix
    t = str(type_val).strip().lower()
    t = t.replace('', '_')
    t = re.sub(r'[^0-9a-zA-Z_]', '', t)
    if t == '':
        t = standard_type_suffix
    return t


def collect_keys_cycle_type(cycle_dates, standard_type_suffix='no_type'):
    y_len = len(cycle_dates['cycles'])
    x_len = len(cycle_dates['cycles'][0]) if len y_len > 0 else 0
    keys_cycle_type = set()
    for y in range(y_len):
        cycle_pixel = cycle_dates['cycles'][y][x]
        if not cycle_pixel:
            continue
        for cycle_info in cycle_pixel:
            cycle_id = cycle_info.get('cycle')
            if cycle_id is None or cycle_id == 0:
                continue
            cycle_type = normalize_type(cycle_info.get('type', standard_type_suffix), standard_type_suffix)
            keys_cycle_type.add((cycle_id, cycle_type))
    return keys_cycle_type


def initialize_arrays(keys_cycle_type, y_len, x_len, no_data_value=-1, out_of_geom_value=-2, outside_mask=None):
    planting_arrays = {}
    harvest_arrays = {}
    for k in keys_cycle_type:
        p = np.full((y_len, x_len), no_data_value, dtype=np.int64)
        h = np.full((y_len, x_len), no_data_value, dtype=np.int64)
        if outside_mask is not None:
            p[outside_mask] = out_of_geom_value
            h[outside_mask] = out_of_geom_value
        planting_arrays = p
        harvest_arrays = h
    return planting_arrays, harvest_arrays


def fill_arrays(cycle_dates, planting_arrays, harvest_arrays, 
                standard_type_suffix='no_type', no_data_value=-1):
    y_len = len(cycle_dates['cycles'])
    x_len = len(cycle_dates['cycles'][0]) if len y_len > 0 else 0
    for y in range(y_len):
        for x in range(x_len):
            cycle_pixel = cycle_dates['cycles'][y][x]
            if not cycle_pixel:
                continue
            for cycle_info in cycle_pixel:
                cycle_id = cycle_info.get('cycle')
                if cycle_id is None or cycle_id == 0:
                    continue
                cycle_type = normalize_type(cycle_info.get('type', standard_type_suffix), standard_type_suffix)
                key = (cycle_id, cycle_type)

                if cycle_type == 'no_cycle':
                    planting_arrays[key][y, x] = no_data_value
                    harvest_arrays[key][y, x] = no_data_value
                else:
                    planting_value = cycle_info.get('planting', no_data_value)
                    harvest_value = cycle_info.get('harvest', no_data_value)

                    if planting_value is None or (isinstance(planting_value, float) and math.isnan(planting_value)):
                        planting_value = no_data_value
                    else:
                        planting_value = int(planting_value)

                    if harvest_value is None or (isinstance(harvest_value, float) and math.isnan(harvest_value)):
                        harvest_value = no_data_value
                    else:
                        harvest_value = int(harvest_value)

                    planting_arrays[key][y, x] = planting_value
                    harvest_arrays[key][y, x] = harvest_value


def define_keys_to_write(planting_arrays, harvest_arrays,
                         ignore_layers_without_data=True,
                         create_pairs=True, no_data_value=-1, out_of_geom_value=-2):
    keys_to_write_planting = set()
    keys_to_write_harvest = set()

    for key in planting_arrays.keys():
        p = planting_arrays[key]
        h = harvest_arrays[key]

        p_valid = (p > 0)
        h_valid = (h > 0)

        if ignore_layers_without_data:
            if create_pairs:
                if np.any(h_valid):
                    keys_to_write_planting.add(key)
                    keys_to_write_harvest.add(key)
            else:
                if np.any(p_valid):
                    keys_to_write_planting.add(key)
                if np.any(h_valid):
                    keys_to_write_harvest.add(key)
        else:
            keys_to_write_planting.add(key)
            keys_to_write_harvest.add(key)

    return keys_to_write_planting, keys_to_write_harvest


def aply_correction(ds, planting_arrays, harvest_arrays, keys_to_write_planting, keys_to_write_harvest,
                    no_data_value=-1, out_of_geom_value=-2):
    

    


