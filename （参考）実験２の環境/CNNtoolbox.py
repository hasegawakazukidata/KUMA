# モジュールのインポート
import psutil
import time
import numpy as np
import pandas as pd
import csv
import os
import optuna
from PIL import Image
from matplotlib import pylab as plt
import tifffile
import warnings
import geopandas as gpd
import random
import glob
from shapely.geometry import box
from rasterio.features import rasterize
from affine import Affine
from optuna.pruners import MedianPruner
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchmetrics.classification import BinaryAUROC
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# --- 3次メッシュコード -> 緯度経度のバウンディングボックス（WGS84） ---
def mesh3_bbox(code: str):
    """
    8桁の3次メッシュコード -> (lon_min, lat_min, lon_max, lat_max) in degrees (EPSG:4326)
    """
    code = str(code)
    if len(code) != 8 or not code.isdigit():
        raise ValueError(f"Invalid 3rd-mesh code: {code} (must be 8 digits)")
    # 第1次メッシュ
    a = int(code[0:2])      # 緯度 40' 単位の番号 = floor(lat * 1.5)
    b = int(code[2:4])      # 経度 1° 単位の番号 = floor(lon - 100)
    # 第2次メッシュの添字
    p = int(code[4])        # 緯度方向 5' のブロック 0-7
    q = int(code[5])        # 経度方向 7.5' のブロック 0-7
    # 第3次メッシュの添字
    r = int(code[6])        # 緯度方向 30" のブロック 0-9
    s = int(code[7])        # 経度方向 45" のブロック 0-9

    # 単位を度に換算
    lat_base = a / 1.5                           # 40' = 2/3°
    lon_base = b + 100
    lat_min = lat_base + (p * (5/60)) + (r * (30/3600))
    lon_min = lon_base + (q * (7.5/60)) + (s * (45/3600))
    lat_max = lat_min + (30/3600)
    lon_max = lon_min + (45/3600)
    return (lon_min, lat_min, lon_max, lat_max)

# 指定した bbox を 256x256 のピクセルグリッドに
def bounds_to_transform(lon_min, lat_min, lon_max, lat_max, width=256, height=256):
    # 左上原点のアフィン（行列は行方向が北→南なので y ピクセルサイズは負）
    xres = (lon_max - lon_min) / width
    yres = (lat_min - lat_max) / height  # negative
    return Affine.translation(lon_min, lat_max) * Affine.scale(xres, yres)

# メッシュ毎 × カテゴリ毎に one-hot でラスタライズ
def build_mesh_tensor(mesh_codes, geojson_path, category_field, categories,
                      width=256, height=256, all_touched=True, dtype=np.uint8):
    """
    Returns: array with shape (num_mesh, height, width, num_categories)
    """
    # 読み込み（CRS は自動検出。なければ WGS84 とみなす）
    gdf = geojson_path
    if gdf.crs is None:
        # 必要に応じて EPSG:4326 を仮定（GeoJSONは通常WGS84）
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)

    # カテゴリ → バンド index
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    out = np.zeros((len(mesh_codes), len(categories), height, width), dtype=dtype)

    for mi, code in enumerate(mesh_codes):
        lon_min, lat_min, lon_max, lat_max = mesh3_bbox(code)
        bbox_poly = box(lon_min, lat_min, lon_max, lat_max)

        # 対象メッシュにかかるフィーチャだけへ絞る（高速化）
        sub = gdf[gdf.geometry.intersects(bbox_poly)]
        if sub.empty:
            continue

        transform = bounds_to_transform(lon_min, lat_min, lon_max, lat_max, width, height)

        # カテゴリごとに 0/1 バンドとして焼く（one-hot）
        for cat, idx in cat_to_idx.items():
            shp = sub[sub[category_field] == cat]
            if shp.empty:
                continue

            shapes = ((geom.intersection(bbox_poly), 1) for geom in shp.geometry if not geom.is_empty)

            band = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                all_touched=all_touched,
                dtype=dtype
            )
            out[mi,idx, :, :] = np.where(band > 0, 1, out[mi,idx, :, :])
    return out

def get_pic_from_image(pic_url):
    pic_temp = np.array(Image.open(pic_url).convert('RGB'))
    return pic_temp[:,:,0], pic_temp[:,:,1], pic_temp[:,:,2]


def get_meshpic(meshid, tiffile):
    x = int(str(meshid)[5]) *10*2*2 + int(str(meshid)[7]) *2*2
    y = int(str(meshid)[4]) *10*2*2 + int(str(meshid)[6]) *2*2 
    x += (int(str(meshid)[2:4])-28)*2*2*10*8
    y += (int(str(meshid)[0:2])-45)*2*2*10*8
    return tiffile[(2*2*10*8 - y - 1 - 8):(2*2*10*8 - y - 1 + 8),(x - 8):(x + 8)]




def _prepare_fold_loggers(fold_n: int):
    trials_csv = os.path.join(log_dir, "optuna_logs/trials.csv")
    if not os.path.exists(trials_csv):
        with open(trials_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trial_number","best_epoch","best_val_auc","best_val_acc",
                "best_test_auc","best_test_acc",
                "lr","weight_decay","batch_size","dropout","base_channels"
            ])
    return log_dir, trials_csv



import pickle
def seve_metrics(name, datalist):
    with open('optuna_pickeles/' + str(name) + '_train_loss_list.pickle', mode='wb') as fo:
        pickle.dump(datalist[0], fo)
    with open('optuna_pickeles/' + str(name) + '_val_loss_list.pickle', mode='wb') as fo:
        pickle.dump(datalist[1], fo)
    with open('optuna_pickeles/' + str(name) + '_test_loss_list.pickle', mode='wb') as fo:
        pickle.dump(datalist[2], fo)
        
    with open('optuna_pickeles/' + str(name) + '_train_acc_list.pickle', mode='wb') as fo:
        pickle.dump(datalist[3], fo)
    with open('optuna_pickeles/' + str(name) + '_val_acc_list.pickle', mode='wb') as fo:
        pickle.dump(datalist[4], fo)
    with open('optuna_pickeles/' + str(name) + '_test_acc_list.pickle', mode='wb') as fo:
        pickle.dump(datalist[5], fo)
 
    with open('optuna_pickeles/' + str(name) + '_train_positive_list.pickle', mode='wb') as fo:
        pickle.dump(datalist[6], fo)
    with open('optuna_pickeles/' + str(name) + '_val_positive_list.pickle', mode='wb') as fo:
        pickle.dump(datalist[7], fo)
    with open('optuna_pickeles/' + str(name) + '_test_positive_list.pickle', mode='wb') as fo:
        pickle.dump(datalist[8], fo)
        
    with open('optuna_pickeles/' + str(name) + '_train_F1_list.pickle', mode='wb') as fo:
        pickle.dump(datalist[9], fo)        
    with open('optuna_pickeles/' + str(name) + '_val_F1_list.pickle', mode='wb') as fo:
        pickle.dump(datalist[10], fo)
    with open('optuna_pickeles/' + str(name) + '_test_F1_list.pickle', mode='wb') as fo:
        pickle.dump(datalist[11], fo)
        
    with open('optuna_pickeles/' + str(name) + '_train_AUC_list.pickle', mode='wb') as fo:
        pickle.dump(datalist[12], fo)        
    with open('optuna_pickeles/' + str(name) + '_val_AUC_list.pickle', mode='wb') as fo:
        pickle.dump(datalist[13], fo)
    with open('optuna_pickeles/' + str(name) + '_test_AUC_list.pickle', mode='wb') as fo:
        pickle.dump(datalist[14], fo)

    with open('optuna_pickeles/' + str(name) + '_train_meanscore_list.pickle', mode='wb') as fo:
        pickle.dump(datalist[15], fo)
    with open('optuna_pickeles/' + str(name) + '_valid_meanscore_list.pickle', mode='wb') as fo:
        pickle.dump(datalist[16], fo)
    with open('optuna_pickeles/' + str(name) + '_test_meanscore_list.pickle', mode='wb') as fo:
        pickle.dump(datalist[17], fo)
    if len(datalist) > 20:
        with open('optuna_pickeles/' + str(name) + '_train_PRAUC_list.pickle', mode='wb') as fo:
            pickle.dump(datalist[18], fo)
        with open('optuna_pickeles/' + str(name) + '_val_PRAUC_list.pickle', mode='wb') as fo:
            pickle.dump(datalist[19], fo)
        with open('optuna_pickeles/' + str(name) + '_test_PRAUC_list.pickle', mode='wb') as fo:
            pickle.dump(datalist[20], fo)