import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def train_lgb(data_dir, num_fold, fold, save_dir):
    train = pd.DataFrame()
    for f in range(num_fold):
        if f != fold:
            tmp = pd.read_csv(os.path.join(data_dir, f"train{f}.csv"))
            train = pd.concat([train, tmp], axis=0)
        else:
            valid = pd.read_csv(os.path.join(data_dir, f"train{f}.csv"))

    train = train.drop(columns=["City"], axis=1)
    train["Country"] = train["Country"].astype("category")
    train["month"] = train["month"].astype("category")
    train["day"] = train["day"].astype("category")
    train["elapsed"] = train["elapsed"].astype("category")

    valid = valid.drop(columns=["City"], axis=1)
    valid["Country"] = valid["Country"].astype("category")
    valid["month"] = valid["month"].astype("category")
    valid["day"] = valid["day"].astype("category")
    valid["elapsed"] = valid["elapsed"].astype("category")

    lgb_train = lgb.Dataset(train.drop(["id", "pm25_mid"], axis=1), train["pm25_mid"])
    lgb_valid = lgb.Dataset(valid.drop(["id", "pm25_mid"], axis=1), valid["pm25_mid"])

    params = {
        "objective": "regression",
        "metric": "rmse",
        "reg_lambda": 0.1,
        "boosting_type": "gbdt",
        "max_depth": 8,
        "num_leaves": int(0.7 * 8 ** 2),
        "learning_rate": 0.03,
        "feature_fraction": 0.9,
        "bagging_freq": 3,
        "bagging_fraction": 0.9,
        "random_state": fold ** 2,
        "verbose": -1,
    }

    model = lgb.train(
        params=params,
        train_set=lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        num_boost_round=150000,
        early_stopping_rounds=100,
    )

    pickle.dump(model, open(os.path.join(save_dir, f"lgb{fold}.pt"), "wb"))

    pred = model.predict(valid.drop(["id", "pm25_mid"], axis=1))
    score = np.sqrt(mean_squared_error(valid["pm25_mid"], pred))
    # print(f"fold:{fold} score: {score}")

    importance = pd.DataFrame(
        model.feature_importance(importance_type="split"),
        index=train.drop(["id", "pm25_mid"], axis=1).columns,
        columns=["importance"],
    )
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(os.path.join(save_dir, f"lgb{fold}_split.csv"))
    importance = pd.DataFrame(
        model.feature_importance(importance_type="gain"),
        index=train.drop(["id", "pm25_mid"], axis=1).columns,
        columns=["importance"],
    )
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(os.path.join(save_dir, f"lgb{fold}_gain.csv"))
    return score


def predict(model_path, test):
    model = pickle.load(open(model_path, "rb"))
    pred = model.predict(test.drop("id", axis=1))
    return pred


def ensemble(test, model_dir, num_fold=5):
    test = test.drop(columns=["City"], axis=1)
    test["Country"] = test["Country"].astype("category")
    test["month"] = test["month"].astype("category")
    test["day"] = test["day"].astype("category")
    test["elapsed"] = test["elapsed"].astype("category")

    pred = []
    for fold in range(num_fold):
        model = pickle.load(open(os.path.join(model_dir, f"lgb{fold}.pt"), "rb"))
        pred_lgb = model.predict(test.drop("id", axis=1).values)
        pred.append(pred_lgb)
    pred = np.mean(pred, axis=0)
    return pred


def main():
    num_fold = 10

    score = []
    for fold in range(num_fold):
        score.append(
            train_lgb(data_dir="../../data/step1", num_fold=num_fold, fold=fold, save_dir="../../output/step1")
        )
    for i, s in enumerate(score):
        print(f"fold{i}: {s:.3f}")
    print(f"avg: {np.mean(score, axis=0):.3f}")

    sub = pd.DataFrame(columns=["id", "judgement"])

    test = pd.read_csv("../../data/step1/test.csv")
    sub["id"] = test["id"]
    pred = ensemble(test, model_dir="../../output/step1/", num_fold=num_fold)
    sub["judgement"] = pred
    print(f'replace_num: {len(sub[sub["judgement"] < 0])}')
    sub.loc[sub["judgement"] < 0, "judgement"] = 0.0
    sub.to_csv("../../output/step1/submission.csv", index=False, header=False)


if __name__ == "__main__":
    main()
