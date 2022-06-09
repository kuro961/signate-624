import datetime
import glob
import os
import re

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from tqdm import tqdm


# 経度をsin, cosにより表現
def lon_norm(df):
    df["lon"] = df["lon"] + 180
    df["lon_cos"] = np.cos(2 * np.pi * df["lon"] / 360)
    df["lon_sin"] = np.sin(2 * np.pi * df["lon"] / 360)
    df = df.drop("lon", axis=1)
    return df


def add_dis_desert(df):
    desert_lat_lon = {
        "sahara": (23.4162, 25.6628),
        "kalahari": (-25.5920, 21.0937),
        "alkhali": (21.0953, 48.7190),
        "gobi": (42.7952, 105.0324),
        "taklamakan": (38.8710, 82.1402),
        "australia": (-24.3897, 131.1084),
        "ameriva": (33.2082, -106.2955),
    }
    for desert in desert_lat_lon.keys():
        dis_list = []
        base = desert_lat_lon[desert]
        for d in df.itertuples():
            dis = geodesic(base, (d.lat, d.lon)).km
            dis_list.append(dis)
        dis_series = pd.Series(dis_list, name=f"{desert}_dis")
        df = pd.concat([df, dis_series], axis=1)
    return df


def serch_close_city(train, test, num=3, include_self=False):
    train_city_groups = train.groupby("City")
    test_city_groups = test.groupby("City")

    # train_city - train_city
    # train_train_close = {都市1: [(都市2, 距離), (都市3, 距離)], ...}
    train_train_close = {}
    for train_city_name1, train_city_group1 in train_city_groups:
        dis_dict = {}
        for train_city_name2, train_city_group2 in train_city_groups:
            dis = geodesic(
                (train_city_group1["lat"].iat[0], train_city_group1["lon"].iat[0]),
                (train_city_group2["lat"].iat[0], train_city_group2["lon"].iat[0]),
            ).km
            dis_dict[train_city_name2] = dis
        sorted_dis_dict = sorted(dis_dict.items(), key=lambda x: x[1])
        train_train_close[train_city_name1] = []
        if include_self:
            for i in range(num):
                train_train_close[train_city_name1].append(sorted_dis_dict[i])
        else:
            for i in range(1, num + 1):
                train_train_close[train_city_name1].append(sorted_dis_dict[i])

    # train_city - test_city
    # test_train_close = {都市1: [(都市2, 距離), (都市3, 距離)], ...}
    test_train_close = {}
    for test_city_name, test_city_group in test_city_groups:
        dis_dict = {}
        for train_city_name, train_city_group in train_city_groups:
            dis = geodesic(
                (test_city_group["lat"].iat[0], test_city_group["lon"].iat[0]),
                (train_city_group["lat"].iat[0], train_city_group["lon"].iat[0]),
            ).km
            dis_dict[train_city_name] = dis
        sorted_dis_dict = sorted(dis_dict.items(), key=lambda x: x[1])
        test_train_close[test_city_name] = []
        for i in range(num):
            test_train_close[test_city_name].append(sorted_dis_dict[i])

    return train_train_close, test_train_close


def add_city_pm25(train, test, train_train_close, test_train_close, data_dir):
    train_city_group = train.groupby("City")
    test_city_group = test.groupby("City")

    # 都市ごとのpm2.5データを追加
    train_city_pm25_data = {}
    for name, group in train_city_group:
        train_city_pm25_data[name] = {}
        train_city_pm25_data[name]["min"] = group["pm25_mid"].min()
        train_city_pm25_data[name]["mid"] = group["pm25_mid"].median()
        train_city_pm25_data[name]["max"] = group["pm25_mid"].max()
        train_city_pm25_data[name]["mean"] = group["pm25_mid"].mean()
        train_city_pm25_data[name]["var"] = group["pm25_mid"].var()
    for name, group in train_city_group:
        for i, (close_city_name, _) in enumerate(train_train_close[name]):
            train.loc[train_city_group.groups[name], [f"close_city{i}_pm25_min"]] = train_city_pm25_data[
                close_city_name
            ]["min"]
            train.loc[train_city_group.groups[name], [f"close_city{i}_pm25_mid"]] = train_city_pm25_data[
                close_city_name
            ]["mid"]
            train.loc[train_city_group.groups[name], [f"close_city{i}_pm25_max"]] = train_city_pm25_data[
                close_city_name
            ]["max"]
            train.loc[train_city_group.groups[name], [f"close_city{i}_pm25_mean"]] = train_city_pm25_data[
                close_city_name
            ]["mean"]
            train.loc[train_city_group.groups[name], [f"close_city{i}_pm25_var"]] = train_city_pm25_data[
                close_city_name
            ]["var"]
            break
    # テストデータには学習データに含まれる近い都市のpm2.5データを追加
    for name, group in test_city_group:
        for i, (close_city_name, _) in enumerate(test_train_close[name]):
            test.loc[test_city_group.groups[name], [f"close_city{i}_pm25_min"]] = train_city_pm25_data[
                close_city_name
            ]["min"]
            test.loc[test_city_group.groups[name], [f"close_city{i}_pm25_mid"]] = train_city_pm25_data[
                close_city_name
            ]["mid"]
            test.loc[test_city_group.groups[name], [f"close_city{i}_pm25_max"]] = train_city_pm25_data[
                close_city_name
            ]["max"]
            test.loc[test_city_group.groups[name], [f"close_city{i}_pm25_mean"]] = train_city_pm25_data[
                close_city_name
            ]["mean"]
            test.loc[test_city_group.groups[name], [f"close_city{i}_pm25_var"]] = train_city_pm25_data[
                close_city_name
            ]["var"]
            break

    # 3日以内のpm2.5データを追加
    # 存在しない場合は、都市ごとのpm2.5データの平均値とする
    print("make train lag data...")
    for name, group in tqdm(train_city_group):
        for i, (close_city_name, dis) in enumerate(train_train_close[name]):
            file_name = re.sub("/", "", close_city_name)
            file_name = re.sub(r"\s+", "_", file_name)
            close_city_df = pd.read_csv(f"{data_dir}/train_{file_name}.csv")
            # 近い都市のpm2.5データを追加
            for j in train_city_group.groups[name]:
                before_lag_data = close_city_df[
                    (close_city_df["elapsed"] == train.iloc[j]["elapsed"] - 1)
                    | (close_city_df["elapsed"] == train.iloc[j]["elapsed"] - 2)
                    | (close_city_df["elapsed"] == train.iloc[j]["elapsed"] - 3)
                ]
                if before_lag_data.empty:
                    train.loc[j, [f"close_city{i}_before_pm25"]] = train_city_pm25_data[close_city_name]["mean"]
                else:
                    train.loc[j, [f"close_city{i}_before_pm25"]] = before_lag_data["pm25_mid"].iloc[0]

    # テストデータには学習データに含まれる近い都市の3日以内のpm2.5データを追加
    # 存在しない場合は、近い都市のpm2.5データの平均値とする
    print("make test lag data...")
    for name, group in tqdm(test_city_group):
        for i, (close_city_name, dis) in enumerate(test_train_close[name]):
            file_name = re.sub("/", "", close_city_name)
            file_name = re.sub(r"\s+", "_", file_name)
            close_city_df = pd.read_csv(f"{data_dir}/train_{file_name}.csv")
            for j in test_city_group.groups[name]:
                before_lag_data = close_city_df[
                    (close_city_df["elapsed"] == test.iloc[j]["elapsed"] - 1)
                    | (close_city_df["elapsed"] == test.iloc[j]["elapsed"] - 2)
                    | (close_city_df["elapsed"] == test.iloc[j]["elapsed"] - 3)
                ]
                if before_lag_data.empty:
                    test.loc[j, [f"close_city{i}_before_pm25"]] = train_city_pm25_data[close_city_name]["mean"]
                else:
                    test.loc[j, [f"close_city{i}_before_pm25"]] = before_lag_data["pm25_mid"].iloc[0]


def day_norm(df):
    elapsed = []
    base = datetime.datetime(year=2019, month=1, day=1)
    for d in df.itertuples():
        # 2019年1月1日を基準に経過日数
        obs_d = datetime.datetime(year=d.year, month=d.month, day=d.day)
        elapsed.append((obs_d - base).days)
    elapsed = pd.Series(elapsed, name="elapsed")
    df = pd.concat([df, elapsed], axis=1)
    df = df.drop(["year"], axis=1)  # 'month', 'day'は残す
    return df


def get_group_kfold(train, save_dir, num_fold=5):
    from sklearn.model_selection import GroupKFold

    group_kfold = GroupKFold(n_splits=num_fold)
    for i, (tr_idx, val_idx) in enumerate(
        group_kfold.split(train.drop("pm25_mid", axis=1), train["pm25_mid"], groups=train["City"])
    ):
        train.loc[val_idx, "fold"] = int(i)
    train["fold"] = train["fold"].astype(np.uint8)
    # train = train.drop(['Country', 'City'], axis=1)

    for f in range(num_fold):
        tmp = train[train["fold"] == f]
        tmp = tmp.drop(["fold"], axis=1)
        save_file = os.path.join(save_dir, f"train{f}.csv")
        tmp.to_csv(save_file, index=False)


def split_data(train, save_dir):
    city_group = train.groupby("City")
    for name, group in city_group:
        file_name = re.sub("/", "", name)
        file_name = re.sub(r"\s+", "_", file_name)
        train.loc[city_group.groups[name]].to_csv(f"{save_dir}/train_{file_name}.csv", index=False)


def rm_split_data(dir_path):
    for file in glob.glob(os.path.join(dir_path, "*.csv")):
        os.remove(file)
