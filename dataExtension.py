import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, f_regression
from xgboost import XGBClassifier, XGBRegressor
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.sampling import Condition
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, r2_score, label_ranking_loss, label_ranking_average_precision_score
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
    for cv_i in range(15):
        # splititng data into train and test
        train_val, test = train_test_split(df, test_size=0.25, random_state=cv_i)
        train, val = train_test_split(train_val, test_size=0.0625, random_state=cv_i)
        train, train_filter, train_target, train_supplement = get_targetFilterAndSupplement(train)
        val, val_filter, val_target, val_supplement = get_targetFilterAndSupplement(val)
        test, test_filter, test_target, test_supplement = get_targetFilterAndSupplement(test)

        #filter data down to the region
        region_train = train.loc[train_filter[region_category] == region]
        region_train_target = train_target.loc[train_filter[region_category] == region]
        region_train_supplement = train_supplement.loc[train_filter[region_category] == region]
        region_train_filter = train_filter.loc[train_filter[region_category] == region]

        region_test = test.loc[test_filter[region_category] == region]
        region_test_target = test_target.loc[test_filter[region_category] == region]
        region_test_supplement = test_supplement.loc[test_filter[region_category] == region]
        region_test_filter = test_filter.loc[test_filter[region_category] == region]

        # define region for getting suplement data
        share_region_train = train.loc[train_filter[region_category] != region]
        share_region_train_target = train_target.loc[train_filter[region_category] != region]
        share_region_test = test.loc[test_filter[region_category] != region]
        share_region_test_target = test_target.loc[test_filter[region_category] != region]

        # data extension - feature engineering
        temp = {"region": region, "cv_i": cv_i, "type": "DE1"}
        pre = PolynomialFeatures(degree=2)
        region_train_poly = pre.fit_transform(region_train)
        region_test_poly = pre.fit_transform(region_test)
        region_train_poly_select = SelectKBest(f_classif, k= int(region_train.shape[1]*1.5)).fit_transform(region_train_poly, region_train_target["tg-default"])
        region_test_poly_select = SelectKBest(f_classif, k= int(region_train.shape[1]*1.5)).fit_transform(region_test_poly, region_test_target["tg-default"])
        clf = XGBClassifier(device="cuda")
        clf.fit(region_train_poly_select, region_train_target["tg-default"])
        preds = clf.predict(region_test_poly_select)
        probs = clf.predict_proba(region_test)[:, 1]
        temp["def_accuracy"] = accuracy_score(region_test_target["tg-default"], preds)
        temp["def_f1"] = f1_score(region_test_target["tg-default"], preds)
        temp["def_demoPar"] = demographic_parity_ratio(region_test_target["tg-default"], preds, sensitive_features= region_test_filter["ethnicity"])
        temp["def_equOdds"] = equalized_odds_ratio(region_test_target["tg-default"], preds, sensitive_features= region_test_filter["ethnicity"])
        white_adv = probs[region_test_filter["ethnicity"] == "non-white"].mean() - probs[region_test_filter["ethnicity"] == "white"].mean()
        white_adv_source = region_test_target["tg-default"][region_test_filter["ethnicity"] == "non-white"].mean() - region_test_target["tg-default"][region_test_filter["ethnicity"] == "white"].mean()
        temp["def_normDiffInPred"] = white_adv / white_adv_source

        clf = XGBClassifier(device="cuda")
        clf.fit(region_train_poly_select, region_train_target["tg-int_rate_cat"])
        preds = clf.predict(region_test_poly_select)
        probs = clf.predict_proba(region_test_poly_select)
        temp["int_accuracy"] = accuracy_score(region_test_target["tg-int_rate_cat"], preds)
        temp["int_f1"] = f1_score(region_test_target["tg-int_rate_cat"], preds, average="macro")
        temp["int_lrl"] = label_ranking_loss(OneHotEncoder().fit_transform(region_test_target["tg-int_rate_cat"].values.reshape(-1, 1)), probs)
        temp["int_lrp"] = label_ranking_average_precision_score(OneHotEncoder().fit_transform(region_test_target["tg-int_rate_cat"].values.reshape(-1, 1)), probs)
        temp["int_demoPar"] = demographic_parity_ratio(region_test_target["tg-int_rate_cat"], preds, sensitive_features=region_test_filter["ethnicity"])


        region_train_poly_select = SelectKBest(f_regression, k= int(region_train.shape[1]*1.5)).fit_transform(region_train_poly, region_train_target["tg-int_rate"])
        region_test_poly_select = SelectKBest(f_regression, k= int(region_train.shape[1]*1.5)).fit_transform(region_test_poly, region_test_target["tg-int_rate"])
        reg = XGBRegressor(device="cuda")
        reg.fit(region_train_poly_select, region_train_target["tg-int_rate"])
        preds = reg.predict(region_test_poly_select)
        temp["int_rmse"] = np.sqrt(mean_squared_error(region_test_target["tg-int_rate"], preds))
        temp["int_r2"] = r2_score(region_test_target["tg-int_rate"], preds)
        white_adv = preds[region_test_filter["ethnicity"] == "non-white"].mean() - preds[region_test_filter["ethnicity"] == "white"].mean()
        white_adv_source = region_test_target["tg-int_rate"][region_test_filter["ethnicity"] == "non-white"].mean() - region_test_target["tg-int_rate"][region_test_filter["ethnicity"] == "white"].mean()
        temp["int_normDiffInPred"] = white_adv / white_adv_source

        results = pd.concat((results, pd.DataFrame(temp, index=[0])))

        # data extension - synthetic data
        temp = {"region": region, "cv_i": cv_i, "type": "DE2"}
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(pd.concat((region_train, region_train_target), axis= 1))
        metadata.update_column(column_name="lc-addr_state", sdtype="categorical")
        synthesizer = CTGANSynthesizer(metadata, cuda= True)
        synthesizer.fit(pd.concat((region_train, region_train_target), axis= 1))
        region_train_synth_cond = [Condition(num_rows= sum(region_train_target["tg-default"] == 0)-sum(region_train_target["tg-default"] == 1), column_values={"tg-default": 1})]
        region_train_synth_comb = synthesizer.sample_from_conditions(conditions= region_train_synth_cond)
        region_train_synth_comb = pd.DataFrame(region_train_synth_comb, columns= list(region_train.columns) + list(region_train_target.columns))
        region_train_target_synth = region_train_synth_comb[["tg-default", "tg-int_rate", "tg-int_rate_cat"]]
        region_train_synth = region_train_synth_comb.drop(columns= ["tg-default", "tg-int_rate", "tg-int_rate_cat"])
        region_train_synth = region_train_synth.astype(dict(zip(list(region_train.columns), region_train.dtypes.reset_index(drop= True).tolist())))
        clf = XGBClassifier(device="cuda")
        clf.fit(pd.concat((region_train, region_train_synth), axis=0), pd.concat((region_train_target["tg-default"], region_train_target_synth["tg-default"]), axis=0))
        preds = clf.predict(region_test)
        temp["accuracy"] = accuracy_score(region_test_target["tg-default"], preds)
        temp["f1"] = f1_score(region_test_target["tg-default"], preds)
        temp["def_demoPar"] = demographic_parity_ratio(region_test_target["tg-default"], preds, sensitive_features=region_test_filter["ethnicity"])
        temp["def_equOdds"] = equalized_odds_ratio(region_test_target["tg-default"], preds, sensitive_features=region_test_filter["ethnicity"])

        target_vc = region_train_target["tg-int_rate_cat"].value_counts()
        region_train_synth_conds = [Condition(num_rows= target_vc.iloc[0] - target_vc[x], column_values={"tg-int_rate_cat": x}) for x in target_vc.index[1:]]
        region_train_synth_comb = synthesizer.sample_from_conditions(conditions=region_train_synth_cond)
        region_train_synth_comb = pd.DataFrame(region_train_synth_comb, columns=list(region_train.columns) + list(region_train_target.columns))
        region_train_target_synth = region_train_synth_comb[["tg-default", "tg-int_rate", "tg-int_rate_cat"]]
        region_train_synth = region_train_synth_comb.drop(columns=["tg-default", "tg-int_rate", "tg-int_rate_cat"])
        region_train_synth = region_train_synth.astype(dict(zip(list(region_train.columns), region_train.dtypes.reset_index(drop=True).tolist())))
        clf = XGBClassifier(device="cuda")
        clf.fit(pd.concat((region_train, region_train_synth), axis=0), pd.concat((region_train_target["tg-int_rate_cat"], region_train_target_synth["tg-int_rate_cat"]), axis=0))
        preds = clf.predict(region_test)
        probs = clf.predict_proba(region_test)
        temp["int_accuracy"] = accuracy_score(region_test_target["tg-int_rate_cat"], preds)
        temp["int_f1"] = f1_score(region_test_target["tg-int_rate_cat"], preds, average="macro")
        temp["int_lrl"] = label_ranking_loss(OneHotEncoder().fit_transform(region_test_target["tg-int_rate_cat"].values.reshape(-1, 1)), probs)
        temp["int_lrp"] = label_ranking_average_precision_score(OneHotEncoder().fit_transform(region_test_target["tg-int_rate_cat"].values.reshape(-1, 1)), probs)
        temp["int_demoPar"] = demographic_parity_ratio(region_test_target["tg-int_rate_cat"], preds, sensitive_features=region_test_filter["ethnicity"])

        region_train_synth_comb = synthesizer.sample(num_rows=region_train.shape[0])
        region_train_synth_comb = pd.DataFrame(region_train_synth_comb, columns=list(region_train.columns) + list(region_train_target.columns))
        region_train_target_synth = region_train_synth_comb[["tg-default", "tg-int_rate", "tg-int_rate_cat"]]
        region_train_synth = region_train_synth_comb.drop(columns=["tg-default", "tg-int_rate", "tg-int_rate_cat"])
        region_train_synth = region_train_synth.astype(dict(zip(list(region_train.columns), region_train.dtypes.reset_index(drop=True).tolist())))
        reg = XGBRegressor(device="cuda")
        reg.fit(pd.concat((region_train, region_train_synth), axis=0), pd.concat((region_train_target["tg-int_rate"], region_train_target_synth["tg-int_rate"]), axis=0))
        preds = reg.predict(region_test)
        temp["rmse"] = np.sqrt(mean_squared_error(region_test_target["tg-int_rate"], preds))
        temp["r2"] = r2_score(region_test_target["tg-int_rate"], preds)
        results = pd.concat((results, pd.DataFrame(temp, index=[0])))
        white_adv = preds[region_test_filter["ethnicity"] == "non-white"].mean() - preds[region_test_filter["ethnicity"] == "white"].mean()
        white_adv_source = region_test_target["tg-int_rate"][region_test_filter["ethnicity"] == "non-white"].mean() - region_test_target["tg-int_rate"][region_test_filter["ethnicity"] == "white"].mean()
        temp["int_normDiffInPred"] = white_adv / white_adv_source

        results = pd.concat((results, pd.DataFrame(temp, index=[0])))

#results = results.groupby(["region", "type"]).mean().reset_index()
#results = results.drop("cv_i", axis=1)
results.to_excel("./results/sub/dataExtension.xlsx", index=False)