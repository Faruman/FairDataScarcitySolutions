import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, r2_score, label_ranking_loss, label_ranking_average_precision_score
from sklearn.base import BaseEstimator
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio


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
for region in tqdm(regions):
    for cv_i in range(30):
        # splititng data into train and test
        train_val, test = train_test_split(df, test_size=0.25, random_state= cv_i)
        train, val = train_test_split(train_val, test_size=0.0625, random_state=cv_i)
        train, train_filter, train_target, train_supplement = get_targetFilterAndSupplement(train)
        val, val_filter, val_target, val_supplement = get_targetFilterAndSupplement(val)
        test, test_filter, test_target, test_supplement = get_targetFilterAndSupplement(test)

        #filter data down to the region
        region_train = train.loc[train_filter[region_category] == region]
        region_train_target = train_target.loc[train_filter[region_category] == region]
        region_train_supplement = train_supplement.loc[train_filter[region_category] == region]
        region_train_filter = train_filter.loc[train_filter[region_category] == region]

        region_val = val.loc[val_filter[region_category] == region]
        region_val_target = val_target.loc[val_filter[region_category] == region]
        region_val_supplement = val_supplement.loc[val_filter[region_category] == region]
        region_val_filter = val_filter.loc[val_filter[region_category] == region]

        region_test = test.loc[test_filter[region_category] == region]
        region_test_target = test_target.loc[test_filter[region_category] == region]
        region_test_supplement = test_supplement.loc[test_filter[region_category] == region]
        region_test_filter = test_filter.loc[test_filter[region_category] == region]

        # define region for getting suplement data
        share_region_train = train.loc[train_filter[region_category] != region]
        share_region_train_target = train_target.loc[train_filter[region_category] != region]
        share_region_test = test.loc[test_filter[region_category] != region]
        share_region_test_target = test_target.loc[test_filter[region_category] != region]

        # active learning
        temp = {"region": region, "cv_i": cv_i, "type": "AL1"}
        n_queries = int(region_val.shape[0] * 0.2)

        learner = ActiveLearner(estimator=XGBClassifier(device="cuda", n_estimators= 500), query_strategy=uncertainty_sampling, X_training=region_train.values, y_training=region_train_target["tg-default"].values)
        temp_region_val = region_val
        temp_region_val_target = region_val_target
        for i in range(int(n_queries/20)):
            query_idx, query_inst = learner.query(temp_region_val.values, n_instances= 20)
            y_new = temp_region_val_target["tg-default"].iloc[query_idx]
            learner.teach(query_inst, y_new.values)
            temp_region_val, temp_region_val_target = temp_region_val.drop(temp_region_val.index[query_idx]), temp_region_val_target.drop(temp_region_val.index[query_idx])
        preds = learner.predict(region_test)
        temp["def_accuracy"] = accuracy_score(region_test_target["tg-default"], preds)
        temp["def_f1"] = f1_score(region_test_target["tg-default"], preds)
        temp["def_demoPar"] = demographic_parity_ratio(region_test_target["tg-default"], preds, sensitive_features=region_test_filter["ethnicity"])
        temp["def_equOdds"] = equalized_odds_ratio(region_test_target["tg-default"], preds, sensitive_features=region_test_filter["ethnicity"])
        temp["def_DiffInPred"] = preds[region_test_filter["ethnicity"] == "non-white"].mean() - preds[region_test_filter["ethnicity"] == "white"].mean()
        temp["def_DiffInSource"] = region_test_target["tg-default"][region_test_filter["ethnicity"] == "non-white"].mean() - region_test_target["tg-default"][region_test_filter["ethnicity"] == "white"].mean()
        temp["def_normDiffInPred"] = temp["def_DiffInPred"] / temp["def_DiffInSource"]

        #learner = ActiveLearner(estimator=XGBClassifier(device="cuda", n_estimators= 500), query_strategy=uncertainty_sampling, X_training=region_train.values, y_training=region_train_target["tg-int_rate_cat"].values)
        #temp_region_val = region_val
        #temp_region_val_target = region_val_target
        #for i in range(int(n_queries/20)):
        #    query_idx, query_inst = learner.query(temp_region_val.values, n_instances= 20)
        #    y_new = temp_region_val_target["tg-int_rate_cat"].iloc[query_idx]
        #    learner.teach(query_inst, y_new.values)
        #    temp_region_val, temp_region_val_target = temp_region_val.drop(temp_region_val.index[query_idx]), temp_region_val_target.drop(temp_region_val.index[query_idx])
        #preds = learner.predict(region_test)
        #probs = learner.predict_proba(region_test)
        #temp["int_accuracy"] = accuracy_score(region_test_target["tg-int_rate_cat"], preds)
        #temp["int_f1"] = f1_score(region_test_target["tg-int_rate_cat"], preds, average="macro")
        #temp["int_lrl"] = label_ranking_loss(OneHotEncoder().fit_transform(region_test_target["tg-int_rate_cat"].values.reshape(-1, 1)), probs)
        #temp["int_lrp"] = label_ranking_average_precision_score(OneHotEncoder().fit_transform(region_test_target["tg-int_rate_cat"].values.reshape(-1, 1)), probs)
        #temp["int_demoPar"] = demographic_parity_ratio(region_test_target["tg-int_rate_cat"], preds, sensitive_features=region_test_filter["ethnicity"])

        class custom_XGBRegressor(XGBRegressor, BaseEstimator):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
            def predict_std(self, X):
                individual_logits = []
                for tree_ in self.get_booster():
                    individual_logits.append(tree_.predict(xgb.DMatrix(X), output_margin=True))
                individual_logits = np.vstack(individual_logits)
                std = np.sqrt(np.var(individual_logits, axis=0))
                xgb_preds = self.get_booster().predict(xgb.DMatrix(X))
                return xgb_preds, std

        def XGBR_regression_std(regressor, X, n_instances):
            _, std = regressor.estimator.predict_std(X)
            query_idx = np.argpartition(std, -n_instances)[-n_instances:]
            return query_idx, X[query_idx]

        learner = ActiveLearner(estimator= custom_XGBRegressor(device="cuda", n_estimators= 500), query_strategy=XGBR_regression_std, X_training=region_train.values, y_training=region_train_target["tg-int_rate"].values)
        temp_region_val = region_val
        temp_region_val_target = region_val_target
        for i in range(int(n_queries/20)):
            query_idx, query_inst = learner.query(temp_region_val.values, n_instances= 20)
            y_new = temp_region_val_target["tg-int_rate"].iloc[query_idx]
            learner.teach(query_inst, y_new.values)
            temp_region_val, temp_region_val_target = temp_region_val.drop(temp_region_val.index[query_idx]), temp_region_val_target.drop(temp_region_val.index[query_idx])
        preds = learner.predict(region_test)
        temp["int_rmse"] = np.sqrt(mean_squared_error(region_test_target["tg-int_rate"], preds))
        temp["int_r2"] = r2_score(region_test_target["tg-int_rate"], preds)
        temp["int_DiffInPred"] = preds[region_test_filter["ethnicity"] == "non-white"].mean() - preds[region_test_filter["ethnicity"] == "white"].mean()
        temp["int_DiffInSource"] = region_test_target["tg-int_rate"][region_test_filter["ethnicity"] == "non-white"].mean() - region_test_target["tg-int_rate"][region_test_filter["ethnicity"] == "white"].mean()
        temp["int_normDiffInPred"] = temp["int_DiffInPred"] / temp["int_DiffInSource"]

        results = pd.concat((results, pd.DataFrame(temp, index=[0])))

#results = results.groupby(["region", "type"]).mean().reset_index()
#results = results.drop("cv_i", axis=1)
results.to_excel("./results/sub/activeLearning.xlsx", index=False)