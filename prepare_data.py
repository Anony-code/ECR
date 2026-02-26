import pandas as pd
import numpy as np
import pickle
import argparse
import sys
import os
# sys.path.append('..')
from utils_raw import load_new_ad, load_new_pd_common, load_psm_data, load_new_pd
# from utils_newdata import load_new_f_v1_with_demo, load_split
from sklearn.model_selection import KFold, StratifiedKFold
import torch
import json
from tqdm import tqdm


def build_examples(df_5, df_early, labels, col2name, dx_cols, med_cols, idx=None, demof=None):
    if idx is None:
        print("Error: idx is None in build_examples")

    out = []

    def map_features(cols, df_row, drop_unmapped_features_across_all, missing_features):
        mapped = []
        for c in cols:
            if df_row[c] <= 0:
                continue
            if c.strip() not in col2name:
                drop_unmapped_features_across_all += 1
                if c.strip() not in missing_features:
                    # print(f"[WARN] missing mapping for feature: {c}")
                    missing_features.add(c)
                continue
            mapped.append(col2name[c])
        return mapped, drop_unmapped_features_across_all, missing_features

    missing_features = set()
    drop_unmapped_features_across_all = 0
    for i in tqdm(idx):
        row5 = df_5.iloc[i]
        # print(row5)
        pid = str(row5["person_id"])

        sex = None
        age = None
        try:
            sex = demof[demof['person_id'] == pid]['sex_str'].values[0]
        except:
            pass
        try:
            age = demof[demof['person_id'] == pid]['age_str'].values[0]
        except:
            pass

        base_dx, drop_unmapped_features_across_all, missing_features = map_features(dx_cols, row5, drop_unmapped_features_across_all, missing_features)
        base_md, drop_unmapped_features_across_all, missing_features = map_features(med_cols, row5, drop_unmapped_features_across_all, missing_features)

        delta_dx, delta_md = [], []
        if df_early is not None and i < len(df_early):
            rowE = df_early.iloc[i]
            delta_dx, _, _ = map_features(
                [c for c in dx_cols if c not in df_5.columns or row5[c] <= 0],
                rowE, drop_unmapped_features_across_all=0, missing_features=set()
            )
            delta_md, _, _ = map_features(
                [c for c in med_cols if c not in df_5.columns or row5[c] <= 0],
                rowE, drop_unmapped_features_across_all=0, missing_features=set()
            )

        out.append({
            "id": pid,
            "base_codes_diagnosis": base_dx,
            "base_codes_medication": base_md,
            "delta_codes_diagnosis": delta_dx,
            "delta_codes_medication": delta_md,
            "label": int(labels[i]),
            "sex": sex,
            "age": int(np.round(float(age))),
        })

    print(
        "|drop_unmapped_features_across_all:",
        drop_unmapped_features_across_all,
        "missing_features:",
        missing_features,
    )
    return out


# python phase0.py --ratio 5 --dataset pd # variale has no effect
# python phase0.py --ratio 10 --dataset ad --use_neg_ratio --neg_ratio 5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--ratio', type=int, default=10)
    parser.add_argument('--variable', type=str, default='visit2')
    parser.add_argument('--use_neg_ratio', action='store_true')
    parser.add_argument('--neg_ratio', type=int, default=4)
    parser.add_argument('--neg_ratio_seed', type=int, default=99)
    parser.add_argument('--dataset', type=str, default='ad')

    args = parser.parse_args()
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print('Args for preparing data:', args)

    ratio = args.ratio
    variable = args.variable
    use_neg_ratio = args.use_neg_ratio
    neg_ratio = args.neg_ratio


    global projdir
    # projdir = '/home/shichao/Peiremote/llm_ad_dec22/llm_main_imp'
    projdir = '.'

    if args.dataset == 'ad':
        f, t , _ = load_new_ad(projdir=projdir)
        f_thisyear = f['CP_1_5_yr']
        t_thisyear= t['CP_1_5_yr']



    elif args.dataset == 'pd':
        f, t , _, empty_patients = load_new_pd(projdir=projdir,  use_f2496=False )
        f_thisyear = f['CP_1_5_yr']
        t_thisyear= t['CP_1_5_yr']


    elif args.dataset == 'adrd':
        f = pickle.load(open(f'{projdir}/ADRD_data/matched_f_portion_common_5_ratio_10.pkl', 'rb'))
        t = pickle.load(open(f'{projdir}/ADRD_data/matched_t_portion_common_5_ratio_10.pkl', 'rb'))

        matched_f_thisyear = f[5]
        matched_t_thisyear = t[5]
        matched_f_thisyear = matched_f_thisyear.rename(columns={
            "HASH_SUBJECT_ID": "person_id",})
        matched_f_thisyear.columns = [i.strip() for i in matched_f_thisyear.columns]
        f2 = pickle.load(open(f'{projdir}/ADRD_data/test_f_portion_common_5.pkl', 'rb'))
        t2 = pickle.load(open(f'{projdir}/ADRD_data/test_t_portion_common_5.pkl', 'rb'))

        test_f_thisyear = f2[5]
        test_t_thisyear = t2[5]
        test_f_thisyear = test_f_thisyear.rename(columns={
            "HASH_SUBJECT_ID": "person_id",})
        test_f_thisyear = test_f_thisyear.reindex(columns=matched_f_thisyear.columns, fill_value=0)
        test_f_thisyear.columns = [i.strip() for i in test_f_thisyear.columns]

        f_thisyear = pd.concat([matched_f_thisyear, test_f_thisyear], axis=0)
        t_thisyear = np.concat([matched_t_thisyear, test_t_thisyear], axis=0)


    label_tensor = torch.tensor(t_thisyear, dtype=torch.long).to(device)
    print('\nLabel tensor:', label_tensor.shape)
    labels = np.array(label_tensor.cpu().numpy(), copy=True)


    use_demo = 1
    if use_demo:
        if args.dataset == 'adrd':
            demo_cols = ['age_at_prediction_window']
            fdemo = f_thisyear[[c for c in ['person_id'] + demo_cols]]
            f_thisyear = f_thisyear[[c for c in f_thisyear.columns if c not in  demo_cols]]
            fdemo["age_str"] = fdemo["age_at_prediction_window"] + 5
            fdemo['person_id'] = fdemo['person_id'].astype(str)

        else:
            demo_cols = ['age_at_prediction_window',
            'sex_at_birth_Male', 'sex_at_birth_Nonbinary/unknown',
            'race_Black or African American',
            'race_Middle Eastern or North African', 'race_Other/unknown',
            'race_White']

            fdemo = f_thisyear[[c for c in ['person_id'] + demo_cols]]
            f_thisyear = f_thisyear[[c for c in f_thisyear.columns if c not in  demo_cols]]

            fdemo["sex_str"] = "female"
            fdemo.loc[fdemo["sex_at_birth_Male"] == 1, "sex_str"] = "male"
            fdemo.loc[fdemo["sex_at_birth_Nonbinary/unknown"] == 1, "sex_str"] = "unknown"

            fdemo["sex_str"] = "female"
            fdemo.loc[fdemo["sex_at_birth_Male"] == 1, "sex_str"] = "male"
            fdemo.loc[fdemo["sex_at_birth_Nonbinary/unknown"] == 1, "sex_str"] = "unknown"
            fdemo["age_str"] = fdemo["age_at_prediction_window"] + 5
            fdemo['person_id'] = fdemo['person_id'].astype(str)
    else:
        demo_cols = []
        fdemo = None
    print('\nBuild demographic data:', fdemo.shape, fdemo.person_id.nunique())


    if args.dataset != 'adrd':
        early_f1y = f['CP_1_1_yr']
        early_f1y = early_f1y[[c for c in early_f1y.columns if c not in demo_cols]]

    else:
        early_f1y = f[1]
        early_f1y = early_f1y.rename(columns={
                "HASH_SUBJECT_ID": "person_id",})
        early_f1y = early_f1y[[c for c in early_f1y.columns if c not in demo_cols]]


    if args.dataset == 'ad':
        col2name = pickle.load(open(f'{projdir}/data_material/all_map.pkl', 'rb'))
        f_this_year_processed_as_reference = pickle.load(open(\
            f'{projdir}/data_material/nov22_f_this_year_processed_final_f_v1_gwas_rare_highfreq1500_year5_fold0.pkl', 'rb'))
        f_this_year_patients_as_reference = f_thisyear['person_id'].reset_index()
        f_this_year_patients_as_reference['person_id'] = f_this_year_patients_as_reference['person_id'].astype(str)
        cols = f_this_year_processed_as_reference.columns.tolist()
        pcols = ['person_id'] + cols
        dx_cols = [c for c in cols if c.endswith('_dx')]
        med_cols = [c for c in cols if c.endswith('_rx')]

    elif args.dataset == 'pd':
        col2name = pickle.load(open(f'{projdir}/data_material/all_map.pkl', 'rb'))
        f_this_year_patients_as_reference =  f_thisyear['person_id'].reset_index()
        f_this_year_patients_as_reference['person_id'] = f_this_year_patients_as_reference['person_id'].astype(str)

        pcols = cols = f_thisyear.columns.tolist()
        dx_cols = [c for c in cols if c.endswith('_dx')]
        med_cols = [c for c in cols if c.endswith('_rx')]

    elif args.dataset == 'adrd':
        col2name = pickle.load(open(f'{projdir}/data_material/all_map.pkl', 'rb'))
        f_this_year_patients_as_reference =  f_thisyear['person_id'].reset_index()
        f_this_year_patients_as_reference['person_id'] = f_this_year_patients_as_reference['person_id'].astype(str)

        pcols = cols = f_thisyear.columns.tolist()
        dx_cols = [c for c in cols if c.endswith('_dx')]
        med_cols = [c for c in cols if c.endswith('_rx')]

    """
    psm_split structure:
    - train_fold{n}: one fold's 4:1 split; index 1 is the hold-out set
    - trmatch_ratio{r}_fold{n}: matched subset inside the training portion
    """
    if args.dataset  != 'adrd':
        psm_split = load_psm_data(ratio=ratio, version=1, variable=variable, dataset=args.dataset)

        print('\nFold:', args.fold)

        psm_match_tr = [str(i) for i in  psm_split[f'trmatch_ratio{ratio}_fold{args.fold}']]
        print('\nLoad PSM match set, size:', len(psm_match_tr), ', ratio:', ratio)

        psm_split_ts = [str(i) for i in  psm_split[f'train_fold{args.fold}'][1]]
        print('Load PSM hold-out set, size:', len(psm_split_ts))

        train_pids_all = [str(i) for i in  psm_split[f'train_fold{args.fold}'][0]] # all_training data
        print('Load all training set, size:', len(train_pids_all))

        if args.dataset == 'pd':
            if empty_patients is not None:
                empty_patients = [str(i) for i in empty_patients]
                psm_match_tr_ = [i for i in psm_match_tr if i not in empty_patients]
                psm_split_ts_ = [i for i in psm_split_ts if i not in empty_patients]
                train_pids_all_ = [i for i in train_pids_all if i not in empty_patients]
                print("Remove empty patients under PD:", '|match set: ', len(psm_match_tr_), '|test set: ', len(psm_split_ts_), '|train set: ', len(train_pids_all_))

        psm_split_linx = f_this_year_patients_as_reference.index[
            f_this_year_patients_as_reference['person_id'].isin(psm_match_tr)
        ].tolist()
        print('Patient indices from this_year_patients (that are within PSM match set), size:', len(psm_split_linx))

        test_idx = f_this_year_patients_as_reference.index[
            f_this_year_patients_as_reference['person_id'].isin(psm_split_ts)
        ].tolist()
        print('Patient indices from this_year_patients (that are within hold-out test set), size:', len(test_idx))

        train_idx_all = f_this_year_patients_as_reference.index[
            f_this_year_patients_as_reference['person_id'].isin(train_pids_all)
        ].tolist()
        print('Patient indices from this_year_patients (that are within all training set), size:', len(train_idx_all))


    elif args.dataset == 'adrd':
        psm_match_tr = matched_f_thisyear.person_id.unique().tolist()
        print('\nLoad PSM match set, size:', len(psm_match_tr), ', ratio:', ratio)

        psm_split_ts = test_f_thisyear.person_id.unique().tolist()
        print('Load PSM hold-out set, size:', len(psm_split_ts))

        psm_split_linx = f_thisyear.index[
            f_thisyear['person_id'].isin(psm_match_tr)
        ].tolist()
        print('Patient indices from this_year_patients (that are within PSM match set), size:', len(psm_split_linx))

        test_idx = f_thisyear.index[
            f_thisyear['person_id'].isin(psm_split_ts)
        ].tolist()
        print('Patient indices from this_year_patients (that are within hold-out test set), size:', len(test_idx))

        train_idx_all, train_pids_all = None, None

    f_this_year = f_thisyear.reindex(columns=pcols, fill_value=0)
    early_f1y   = early_f1y.reindex(columns=pcols, fill_value=0)
    f_this_year['person_id'] = f_this_year['person_id'].astype(str)
    early_f1y['person_id'] = early_f1y['person_id'].astype(str)
    print('\tReindexing f_this_year and early_f1y to have same columns as reference:', len(pcols))

    if args.dataset == 'ad' or args.dataset == 'pd' or args.dataset == 'adrd':
        # if matched
        if args.dataset == 'adrd':
            train_idx = [int(i) for i in psm_split_linx]
        else:
            train_idx = [int(i) for i in train_idx_all if i in  psm_split_linx]


        print('\nSplit train', len(train_idx), train_idx[:5], train_idx[-5:])
        print('Split test', len(test_idx), test_idx[:5], test_idx[-5:])

        if use_neg_ratio:
            print('\nUse neg ratio......')
            pos_idx = [i for i in train_idx if labels[i] == 1]
            neg_idx = [i for i in train_idx if labels[i] == 0]

            rng = np.random.default_rng(args.neg_ratio_seed)

            target_neg = min(len(neg_idx), len(pos_idx) * neg_ratio)
            neg_keep = rng.choice(neg_idx, size=target_neg, replace=False).tolist() if target_neg > 0 else []
            train_idx = pos_idx + neg_keep
            train_idx = sorted(train_idx)
            print('|Sampled training (sorted): ', len(train_idx), train_idx[:5], train_idx[-5:])
            print(
                f'|Applied neg subsampling: pos={len(pos_idx)}, '
                f'|neg_keep={len(neg_keep)}, neg_ratio=1:{neg_ratio}'
            )
        else:
            print('\nUse NO neg ratio......')

        print('Really used:', 'train_idx:', len(train_idx), 'test_idx:', len(test_idx))

        if args.dataset == 'pd':
            print('\nAdd a few mapping from rxcui to drug name:', len(col2name) )
            add_map = {'11476_rx':'tiludronic acid', '1364479_rx':'lomitapide',
                    '183379_rx':'rivastigmine', '310436_rx':'galantamine',  '996561_rx':'memantine'}
            col2name = col2name | add_map
            print('Add a few mapping from rxcui to drug name:', len(add_map) , '-->', + len(col2name))

        if args.dataset == 'adrd':
            print('\nInitial mapping:', len(col2name) )

            add_map_rxcui = {'11476_rx':'tiludronic acid', '1364479_rx':'lomitapide',
                    '183379_rx':'rivastigmine', '310436_rx':'galantamine',  '996561_rx':'memantine'}

            invalid_mapping = {'1010_dx':'Other tests',
                               '962.3_dx':'Hormones and synthetic substitutes causing adverse effects in therapeutic use',
                               '1010.3_dx': 'Screening for other diseases and disorders',
                               '1014_dx': 'Effects of heat, cold and air pressure',
                               '339_dx':"Other headache syndromes",
                               '1005_dx':"Other symptoms",
                               '1100_dx':"Family history",
                               '1010.2_dx':"Screening for malignant neoplasms",
                               '1010.5_dx':'',
                               '1000_dx':'',
                               '1001_dx':'',
                               '1002_dx':'',
                               '1003_dx':'',
                               '1004_dx':'',
                               '1005_dx':'',
                               '1006_dx':'',
                               '1007_dx':'',
                               '1008_dx':'',
                               '1010_dx':'',
                               '1011_dx':'',
                               '1012_dx':'',
                               '1013_dx':'',
                               '1014_dx':'',
                               '1015_dx':'',
                               '1016_dx':'',
                               }

            col2name = col2name | add_map_rxcui
            print('Add a few mapping from rxcui-to-drug-name and from phecode-to-phename:',  len(add_map_rxcui), '-->', + len(col2name))

            for k in invalid_mapping:
                col2name.pop(k, None)
            print('Drop invalid mapping:', len(invalid_mapping), '-->', + len(col2name))

        print('\nBuilding training examples...')
        examples = build_examples(
            df_5=f_this_year,
            df_early=early_f1y,
            labels=labels,
            col2name=col2name,
            dx_cols=dx_cols,
            med_cols=med_cols,
            idx=train_idx,
            demof=fdemo
        )


        out_dir = "data"
        os.makedirs(out_dir, exist_ok=True)

        if args.dataset == 'ad':
            neg_tag = f"_neg{neg_ratio}" if use_neg_ratio else ""
            with open(f"data/AD_train_fold{args.fold}_ratio{ratio}{neg_tag}_{variable}.jsonl", "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")
        elif args.dataset == 'pd':
            neg_tag = f"_neg{neg_ratio}" if use_neg_ratio else ""
            with open(f"data/PD_train_fold{args.fold}_ratio{ratio}{neg_tag}.jsonl", "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")
        elif args.dataset == 'adrd':
            neg_tag = f"_neg{neg_ratio}" if use_neg_ratio else ""
            with open(f"data/ADRD_train_fold{args.fold}_ratio{ratio}{neg_tag}.jsonl", "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")


        print('\nBuilding testing examples...')
        examples_test = build_examples(
            df_5=f_this_year,
            df_early=None,
            labels=labels,
            col2name=col2name,
            dx_cols=dx_cols,
            med_cols=med_cols,
            idx=test_idx,
            demof=fdemo
        )

        if args.dataset == 'ad':
            # ratio = 10, neg ratio can be 5
            with open(f"data/AD_test_fold{args.fold}_ratio{ratio}{neg_tag}_{variable}.jsonl", "w") as f:
                for ex in examples_test:
                    f.write(json.dumps(ex) + "\n")
        elif args.dataset == 'pd':
            # ratio = 5, neg ratio can be 3
            with open(f"data/PD_test_fold{args.fold}_ratio{ratio}{neg_tag}.jsonl", "w") as f:
                for ex in examples_test:
                    f.write(json.dumps(ex) + "\n")
        elif args.dataset == 'adrd':
            # ratio = 10, neg ratio can be 5
            with open(f"data/ADRD_test_fold{args.fold}_ratio{ratio}{neg_tag}.jsonl", "w") as f:
                for ex in examples_test:
                    f.write(json.dumps(ex) + "\n")


if __name__ == "__main__":
    main()
