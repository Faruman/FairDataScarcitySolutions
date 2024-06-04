import random
import json
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, r2_score, label_ranking_loss, label_ranking_average_precision_score
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio
privileged_groups = [{"ethnicity": "white"}]
unprivileged_groups = [{"ethnicity": "black"}]

import seaborn as sns


def get_targetFilterAndSupplement(df, filter_cols: list = ["year", "topregion", "region", "subregion", "ethnicity", "cs-white_ratio", "cs-non-white_ratio"],
                                  target_cols: list = ["tg-default", "tg-int_rate", "tg-int_rate_cat"],
                                  supplement_cols: list = ["hg-housing_index", "ec-gdp_growth", "ec-unemployment", "ec-inflation"]):
    filter = df[filter_cols]
    df = df.drop(filter_cols, axis=1)
    target = df[target_cols]
    df = df.drop(target_cols, axis=1)
    supplement = df[supplement_cols]
    df = df.drop(supplement_cols, axis=1)
    return df, filter, target, supplement

def aggregate(bst_prev, bst_curr):
    """Conduct bagging aggregation for given trees."""
    if not bst_prev:
        return bst_curr

    # Get the tree numbers
    tree_num_prev, _ = _get_tree_nums(bst_prev)
    _, paral_tree_num_curr = _get_tree_nums(bst_curr)

    bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
        "num_trees"
    ] = str(tree_num_prev + paral_tree_num_curr)
    iteration_indptr = bst_prev["learner"]["gradient_booster"]["model"][
        "iteration_indptr"
    ]
    bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(
        iteration_indptr[-1] + paral_tree_num_curr
    )

    # Aggregate new trees
    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
    for tree_count in range(paral_tree_num_curr):
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count
        bst_prev["learner"]["gradient_booster"]["model"]["trees"].append(
            trees_curr[tree_count]
        )
        bst_prev["learner"]["gradient_booster"]["model"]["tree_info"].append(0)
    return bst_prev

def _get_tree_nums(xgb_model):
    # Get the number of trees
    tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_trees"
        ]
    )
    # Get the number of parallel trees
    paral_tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_parallel_tree"
        ]
    )
    return tree_num, paral_tree_num


# Load the data
df = pd.read_csv("./data/loanDataPreprocessed.csv")
_, filter, target, supplement = get_targetFilterAndSupplement(df)

# remove regions with too little samples
region_category = "subregion"
regions = filter.groupby([region_category, "ethnicity"]).count()["year"].reset_index()
regions = regions.loc[regions["year"] > 100][region_category].value_counts()
regions = list(regions.loc[regions > 1].index)


# train model for different regions
results = pd.DataFrame()
for cv_i in tqdm(range(15)):
    # splititng data into train and test
    train_val, test = train_test_split(df, test_size=0.25, random_state=cv_i)
    train, val = train_test_split(train_val, test_size=0.0625, random_state=cv_i)
    train, train_filter, train_target, train_supplement = get_targetFilterAndSupplement(train)
    val, val_filter, val_target, val_supplement = get_targetFilterAndSupplement(val)
    test, test_filter, test_target, test_supplement = get_targetFilterAndSupplement(test)

    num_exRounds = 10
    shared_model = None
    region_models = {}
    #init_idx = np.random.randint(train.shape[0], size=500)
    #shared_model.fit(train.iloc[init_idx], train_target["tg-default"].iloc[init_idx])
    #shared_model.fit(pd.DataFrame(np.random.rand(2,train.shape[1]), columns= train.columns), pd.Series([0, 1]))
    #shared_model_xgb_params = shared_model.get_xgb_params()
    for i in range(num_exRounds):
        shared_boosters = []
        init_shared_model = []
        for region_i, region in enumerate(regions):
            # filter data down to the region
            region_train = train.loc[train_filter[region_category] == region]
            region_train_target = train_target.loc[train_filter[region_category] == region]
            region_train_supplement = train_supplement.loc[train_filter[region_category] == region]
            region_train_filter = train_filter.loc[train_filter[region_category] == region]

            #region_model = XGBClassifier(n_estimators= int((100 / num_exRounds) * (i + 1)))
            region_model = XGBClassifier(device= "cuda")
            if shared_model:
                region_model.fit(region_train, region_train_target["tg-default"], xgb_model= shared_model)
            else:
                region_model.fit(region_train, region_train_target["tg-default"])
                init_shared_model.append(copy.deepcopy(region_model))

            region_model_bst = json.loads(bytearray(region_model.get_booster().save_raw("json")))

            #region_model_bst_bytes = bytes(region_model_bst)
            #shared_boosters.append(region_model_bst_bytes)
            shared_boosters.append(region_model_bst)
            region_models[region] = copy.deepcopy(region_model)

            # For testing
            #region_model.get_booster().load_model(bytearray(bytes(json.dumps(region_model_bst), "utf-8")))

        #combine individual xgboost models into share model
        if shared_model:
            shared_model_bst = json.loads(bytearray(shared_model.get_booster().save_raw("json")))
        else:
            init_idx = random.randint(0, len(init_shared_model) - 1)
            shared_model = init_shared_model[init_idx]
            shared_model_bst = json.loads(bytearray(shared_model.get_booster().save_raw("json")))
            shared_boosters.pop(init_idx)

        for shared_booster in shared_boosters:
            shared_model_bst = aggregate(shared_model_bst, shared_booster)

        # Load global model into booster
        shared_model.get_booster().load_model(bytearray(bytes(json.dumps(shared_model_bst), "utf-8")))
    for region in regions:
        #filter data down to the region
        region_test = test.loc[test_filter[region_category] == region]
        region_test_target = test_target.loc[test_filter[region_category] == region]
        region_test_supplement = test_supplement.loc[test_filter[region_category] == region]
        region_test_filter = test_filter.loc[test_filter[region_category] == region]

        # create baseline model
        temp = {"region": region, "cv_i": cv_i, "type": "FL1"}
        preds = region_models[region].predict(region_test)
        probs = region_models[region].predict_proba(region_test)[:, 1]
        temp["def_accuracy"] = accuracy_score(region_test_target["tg-default"], preds)
        temp["def_f1"] = f1_score(region_test_target["tg-default"], preds)
        temp["def_demoPar"] = demographic_parity_ratio(region_test_target["tg-default"], preds, sensitive_features= region_test_filter["ethnicity"])
        temp["def_equOdds"] = equalized_odds_ratio(region_test_target["tg-default"], preds, sensitive_features= region_test_filter["ethnicity"])
        white_adv = probs[region_test_filter["ethnicity"] == "non-white"].mean() - probs[region_test_filter["ethnicity"] == "white"].mean()
        white_adv_source = region_test_target["tg-default"][region_test_filter["ethnicity"] == "non-white"].mean() - region_test_target["tg-default"][region_test_filter["ethnicity"] == "white"].mean()
        temp["def_normDiffInPred"] = white_adv / white_adv_source

        results = pd.concat((results, pd.DataFrame(temp, index=[0])))

    num_exRounds = 10
    shared_model = None
    region_models = {}
    for i in range(num_exRounds):
        shared_boosters = []
        init_shared_model = []
        for region_i, region in enumerate(regions):
            # filter data down to the region
            region_train = train.loc[train_filter[region_category] == region]
            region_train_target = train_target.loc[train_filter[region_category] == region]
            region_train_supplement = train_supplement.loc[train_filter[region_category] == region]
            region_train_filter = train_filter.loc[train_filter[region_category] == region]

            #region_model = XGBClassifier(n_estimators= int((100 / num_exRounds) * (i + 1)))
            region_model = XGBClassifier(device= "cuda")
            if shared_model:
                region_model.fit(region_train, region_train_target["tg-int_rate_cat"], xgb_model= shared_model)
            else:
                region_model.fit(region_train, region_train_target["tg-int_rate_cat"])
                init_shared_model.append(copy.deepcopy(region_model))

            region_model_bst = json.loads(bytearray(region_model.get_booster().save_raw("json")))

            #region_model_bst_bytes = bytes(region_model_bst)
            #shared_boosters.append(region_model_bst_bytes)
            shared_boosters.append(region_model_bst)
            region_models[region] = copy.deepcopy(region_model)

            # For testing
            #region_model.get_booster().load_model(bytearray(bytes(json.dumps(region_model_bst), "utf-8")))

        #combine individual xgboost models into share model
        if shared_model:
            shared_model_bst = json.loads(bytearray(shared_model.get_booster().save_raw("json")))
        else:
            init_idx = random.randint(0, len(init_shared_model) - 1)
            shared_model = init_shared_model[init_idx]
            shared_model_bst = json.loads(bytearray(shared_model.get_booster().save_raw("json")))
            shared_boosters.pop(init_idx)

        for shared_booster in shared_boosters:
            shared_model_bst = aggregate(shared_model_bst, shared_booster)

        # Load global model into booster
        shared_model.get_booster().load_model(bytearray(bytes(json.dumps(shared_model_bst), "utf-8")))
    for region in regions:
        #filter data down to the region
        region_test = test.loc[test_filter[region_category] == region]
        region_test_target = test_target.loc[test_filter[region_category] == region]
        region_test_supplement = test_supplement.loc[test_filter[region_category] == region]
        region_test_filter = test_filter.loc[test_filter[region_category] == region]

        # create baseline model
        temp = {"region": region, "cv_i": cv_i, "type": "FL1"}
        preds = region_models[region].predict(region_test)
        probs = region_models[region].predict_proba(region_test)
        temp["int_accuracy"] = accuracy_score(region_test_target["tg-int_rate_cat"], preds)
        temp["int_f1"] = f1_score(region_test_target["tg-int_rate_cat"], preds, average="macro")
        temp["int_lrl"] = label_ranking_loss(OneHotEncoder().fit_transform(region_test_target["tg-int_rate_cat"].values.reshape(-1, 1)), probs)
        temp["int_lrp"] = label_ranking_average_precision_score(OneHotEncoder().fit_transform(region_test_target["tg-int_rate_cat"].values.reshape(-1, 1)), probs)
        temp["int_demoPar"] = demographic_parity_ratio(region_test_target["tg-int_rate_cat"], preds, sensitive_features=region_test_filter["ethnicity"])

        results = pd.concat((results, pd.DataFrame(temp, index=[0])))

    num_exRounds = 10
    shared_model = None
    region_models = {}
    for i in range(num_exRounds):
        shared_boosters = []
        init_shared_model = []
        for region_i, region in enumerate(regions):
            # filter data down to the region
            region_train = train.loc[train_filter[region_category] == region]
            region_train_target = train_target.loc[train_filter[region_category] == region]
            region_train_supplement = train_supplement.loc[train_filter[region_category] == region]
            region_train_filter = train_filter.loc[train_filter[region_category] == region]

            # region_model = XGBClassifier(n_estimators= int((100 / num_exRounds) * (i + 1)))
            region_model = XGBRegressor(device="cuda")
            if shared_model:
                region_model.fit(region_train, region_train_target["tg-int_rate"], xgb_model=shared_model)
            else:
                region_model.fit(region_train, region_train_target["tg-int_rate"])
                init_shared_model.append(copy.deepcopy(region_model))

            region_model_bst = json.loads(bytearray(region_model.get_booster().save_raw("json")))

            # region_model_bst_bytes = bytes(region_model_bst)
            # shared_boosters.append(region_model_bst_bytes)
            shared_boosters.append(region_model_bst)
            region_models[region] = copy.deepcopy(region_model)

            # For testing
            # region_model.get_booster().load_model(bytearray(bytes(json.dumps(region_model_bst), "utf-8")))

        # combine individual xgboost models into share model
        if shared_model:
            shared_model_bst = json.loads(bytearray(shared_model.get_booster().save_raw("json")))
        else:
            init_idx = random.randint(0, len(init_shared_model) - 1)
            shared_model = init_shared_model[init_idx]
            shared_model_bst = json.loads(bytearray(shared_model.get_booster().save_raw("json")))
            shared_boosters.pop(init_idx)

        for shared_booster in shared_boosters:
            shared_model_bst = aggregate(shared_model_bst, shared_booster)

        # Load global model into booster
        shared_model.get_booster().load_model(bytearray(bytes(json.dumps(shared_model_bst), "utf-8")))
    for region in regions:
        # filter data down to the region
        region_test = test.loc[test_filter[region_category] == region]
        region_test_target = test_target.loc[test_filter[region_category] == region]
        region_test_supplement = test_supplement.loc[test_filter[region_category] == region]
        region_test_filter = test_filter.loc[test_filter[region_category] == region]

        # create baseline model
        temp = {"region": region, "cv_i": cv_i, "type": "FL1"}
        preds = region_models[region].predict(region_test)
        temp["int_rmse"] = np.sqrt(mean_squared_error(region_test_target["tg-int_rate"], preds))
        temp["int_r2"] = r2_score(region_test_target["tg-int_rate"], preds)
        white_adv = preds[region_test_filter["ethnicity"] == "non-white"].mean() - preds[region_test_filter["ethnicity"] == "white"].mean()
        white_adv_source = region_test_target["tg-int_rate"][region_test_filter["ethnicity"] == "non-white"].mean() - region_test_target["tg-int_rate"][region_test_filter["ethnicity"] == "white"].mean()
        temp["int_normDiffInPred"] = white_adv / white_adv_source

        results = pd.concat((results, pd.DataFrame(temp, index=[0])))

#results = results.groupby(["region", "type"]).mean().reset_index()
#results = results.drop("cv_i", axis=1)
results.to_excel("./results/sub/federatedLearning.xlsx", index=False)