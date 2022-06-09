import os
import sys

import pandas as pd
import xfeat

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import preprocess_func as pf


def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def preprocess(train, test, num_fold=5):
    train = pf.day_norm(train)
    test = pf.day_norm(test)

    # pm2.5データを付与
    print("=== add pm2.5 data ===")
    make_dir("../../data/step1")
    pf.split_data(train, save_dir="../../data/step1")
    close_city_num = 3
    train_train_close, test_train_close = pf.serch_close_city(train, test, num=close_city_num, include_self=True)
    pf.add_city_pm25(train, test, train_train_close, test_train_close, "../../data/step1")
    pf.rm_split_data("../../data/step1")

    # 砂漠との距離を付与
    # print('add distance from the desert')
    # train = pf.add_dis_desert(train)
    # test = pf.add_dis_desert(test)

    # 経度をsin, cosで表現
    train = pf.lon_norm(train)
    test = pf.lon_norm(test)

    # Countryをラベルエンコード
    le = xfeat.LabelEncoder(output_suffix="")
    train = pd.concat([train.drop(columns=["Country"], axis=1), le.fit_transform(train[["Country"]])], axis=1)
    test = pd.concat([test.drop(columns=["Country"], axis=1), le.transform(test[["Country"]])], axis=1)

    save_dir = "../../data/step1"
    make_dir(save_dir)
    pf.get_group_kfold(train, save_dir=save_dir, num_fold=num_fold)
    test.to_csv(f"{save_dir}/test.csv", index=False)


def main():
    train = pd.read_csv("../../download_data/train.csv")
    test = pd.read_csv("../../download_data/test.csv")
    preprocess(train, test, num_fold=10)


if __name__ == "__main__":
    main()
