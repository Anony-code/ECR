import pickle

import numpy as np
import pandas as pd


DEMO_COLS = [
    "age_at_prediction_window",
    "sex_at_birth_Male",
    "sex_at_birth_Nonbinary/unknown",
    "race_Black or African American",
    "race_Middle Eastern or North African",
    "race_Other/unknown",
    "race_White",
]


def _load_col2mapped_features(projdir=None):
    if projdir is None:
        f2496 = pd.read_csv("data_material/col2mapped.txt", sep="\t", header=None)
    else:
        f2496 = pd.read_csv(f"{projdir}/data_material/col2mapped.txt", sep="\t", header=None)

    f2496.columns = ["feature", "fname"]
    return [str(i).strip() for i in f2496.feature.tolist()]


def _strip_columns(df):
    cols = [str(i).strip() for i in df.columns.tolist()]
    df.columns = cols
    return df


def load_new_pd_common(projdir=None, use_f2496=False):
    f2496 = _load_col2mapped_features(projdir=projdir)
    demo_cols = DEMO_COLS

    feature_temp = pickle.load(open(f"{projdir}/PD_data/v1ft.pkl", "rb"))

    all_f_v = feature_temp[0]
    all_t_v = feature_temp[1]
    df_aligneds = {}
    empty_patients = []

    for i, j in all_f_v.items():
        j = _strip_columns(j)

        if use_f2496:
            df_aligned = j.reindex(columns=["person_id"] + f2496 + demo_cols, fill_value=0)
        else:
            base_cols = [c for c in j.columns if c not in (["person_id"] + demo_cols)]
            df_aligned = j.reindex(columns=["person_id"] + base_cols + demo_cols, fill_value=0)

        print("\ndf aligned shape:", i, df_aligned.shape)
        zero_sum_cols = df_aligned.columns[df_aligned.sum() == 0]
        print("\tzero_sum_cols", len(zero_sum_cols))

        thres = 50
        print("remove columns with only 1 or 2 individuals...")
        keep_base_cols = [c for c in base_cols if df_aligned[c].sum() > thres]
        df_aligned = df_aligned[["person_id"] + keep_base_cols + demo_cols]
        print("\tresulting in:", df_aligned.shape)

        if keep_base_cols:
            print("remove patients with all 0 in base cols...")
            empty_mask = df_aligned[keep_base_cols].sum(axis=1) == 0
            empty_idx = df_aligned.index[empty_mask].tolist()
            empty_pids = df_aligned.loc[empty_idx, "person_id"].tolist()
            print("\tremove patients with all 0 in base cols:", empty_idx, empty_pids)
        else:
            empty_idx = []
            empty_pids = []

        df_aligned = df_aligned.drop(index=empty_idx)
        empty_patients.extend(empty_pids)
        df_aligneds[i] = df_aligned

    empty_patients = list(set(empty_patients))
    print("empty_patients:", len(empty_patients))
    return df_aligneds, all_t_v, zero_sum_cols, empty_patients


def load_new_pd(projdir=None, use_f2496=False):
    f2496 = _load_col2mapped_features(projdir=projdir)
    demo_cols = DEMO_COLS

    feature_temp = pickle.load(open(f"{projdir}/PD_data/v1ft.pkl", "rb"))

    all_f_v = feature_temp[0]
    all_t_v = feature_temp[1]
    df_aligneds = {}
    empty_patients = []

    for i, j in all_f_v.items():
        j = _strip_columns(j)

        if use_f2496:
            df_aligned = j.reindex(columns=["person_id"] + f2496 + demo_cols, fill_value=0)
        else:
            base_cols = [c for c in j.columns if c not in (["person_id"] + demo_cols)]
            df_aligned = j.reindex(columns=["person_id"] + base_cols + demo_cols, fill_value=0)

        print("\ndf aligned shape:", i, df_aligned.shape)
        zero_sum_cols = df_aligned.columns[df_aligned.sum() == 0]
        print("\tzero_sum_cols", len(zero_sum_cols))

        df_aligneds[i] = df_aligned

        print("remove patients with all 0 in base cols...")
        empty_mask = df_aligned.sum(axis=1) == 0
        empty_idx = df_aligned.index[empty_mask].tolist()
        empty_pids = df_aligned.loc[empty_idx, "person_id"].tolist()
        print("\tremove patients with all 0 in base cols:", empty_idx, empty_pids)
        empty_patients.extend(empty_pids)

        df_aligneds[i] = df_aligned

    empty_patients = list(set(empty_patients))
    print("empty_patients:", len(empty_patients))
    return df_aligneds, all_t_v, zero_sum_cols, empty_patients


def load_new_ad(projdir=None):
    f2496 = _load_col2mapped_features(projdir=projdir)
    demo_cols = DEMO_COLS

    if projdir is None:
        feature_temp = pickle.load(open("AD_data/v1ft.pkl", "rb"))
    else:
        feature_temp = pickle.load(open(f"{projdir}/AD_data/v1ft.pkl", "rb"))

    all_f_v = feature_temp[0]
    all_t_v = feature_temp[1]
    df_aligneds = {}

    for i, j in all_f_v.items():
        j = _strip_columns(j)
        df_aligned = j.reindex(columns=["person_id"] + f2496 + demo_cols, fill_value=0)
        zero_sum_cols = df_aligned.columns[df_aligned.sum() == 0]
        df_aligneds[i] = df_aligned
        print("\tzero_sum_cols", len(zero_sum_cols))

    return df_aligneds, all_t_v, zero_sum_cols


def load_psm_data(ratio, version, variable, dataset="ad", projdir=None):
    psm_path = None
    if version == 1:
        if dataset == "ad":
            if variable is None:
                psm_path = f"{projdir}/AD_data/psm_match_v1.pkl"
            else:
                psm_path = f"{projdir}/AD_data/psm_match_v1_{variable}.pkl"
        elif dataset == "pd":
            if projdir is not None:
                psm_path = f"{projdir}/PD_data/psm_match_v1_visit5_N6.pkl"
            else:
                psm_path = "PD_data/psm_match_v1_visit5_N6.pkl"
            variable = ""

    if psm_path is None:
        raise ValueError(f"Unsupported version/dataset: version={version}, dataset={dataset}")

    print(
        f"[load_psm_data] loading psm: {psm_path} "
        f"(ratio={ratio}, version={version}, variable={variable}, dataset={dataset})"
    )
    psm = pickle.load(open(psm_path, "rb"))

    psm_folds = {}
    for i in range(5):
        psm_folds[i] = psm[f"ratio{ratio}_fold{i}"]
        psm_folds[f"train_fold{i}"] = psm[f"fold{i}_tr_ts"]

        psm_ids = psm_folds[i].index.values.tolist()
        base_ids_len = len(psm_ids)
        for j in range(ratio):
            col = f"psm_control_{j + 1}"
            psm_ids.extend(psm_folds[i][col].values.tolist())

        psm_ids_set = set(psm_ids)
        tr_all = psm[f"fold{i}_tr_ts"][0]
        ts_all = psm[f"fold{i}_tr_ts"][1]
        tr_match = [tr for tr in tr_all if tr in psm_ids_set]

        psm_folds[f"trmatch_ratio{ratio}_fold{i}"] = tr_match
        print(
            f"[load_psm_data] fold{i}: psm_rows={len(psm_folds[i])}, "
            f"train_all={len(tr_all)}, test_all={len(ts_all)}, "
            f"case_ids={base_ids_len}, cases+matched_ctrls={len(psm_ids_set)}, "
            f"train_match={len(tr_match)} ({len(tr_match) / max(1, len(tr_all)):.3f})",
        )

    return psm_folds
