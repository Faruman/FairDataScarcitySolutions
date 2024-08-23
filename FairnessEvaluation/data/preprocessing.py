import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.preprocessing import LabelEncoder, StandardScaler, KBinsDiscretizer
import pickle

def get_us_region(state_code):
       # Define the regions with the corresponding state codes
       regions = {
              "Northeast": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
              "Midwest": ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"],
              "South": ["DE", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"],
              "West": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"]
       }
       # Iterate through the regions to find the state code
       for region, states in regions.items():
              if state_code in states:
                     return region
       return "Unknown"

def get_us_econ_region(state_code):
       # Define the regions with the corresponding state codes
         regions = {
              "New England": ["CT", "ME", "MA", "NH", "RI", "VT"],
              "Mideast": ["NJ", "NY", "PA", "DE", "MD", "DC"],
              "Great Lakes": ["IL", "IN", "MI", "OH", "WI"],
              "Plains": ["IA", "KS", "MN", "MO", "NE", "ND", "SD"],
              "Southeast": ["AL", "AR", "FL", "GA", "KY", "LA", "MS", "NC", "SC", "TN", "VA", "WV"],
              "Southwest": ["AZ", "NM", "OK", "TX"],
              "Rocky Mountain": ["CO", "ID", "MT", "UT", "WY"],
              "Far West": ["AK", "CA", "HI", "NV", "OR", "WA"],
         }
         # Iterate through the regions to find the state code
         for region, states in regions.items():
                  if state_code in states:
                        return region
         return "Unknown"


census_dict = {
       "Estimate!!Total": "total",
       "Estimate!!Total!!White alone": "white",
       "Estimate!!Total!!Black or African American alone": "black",
       "Estimate!!Total!!American Indian and Alaska Native alone": "native",
       "Estimate!!Total!!Asian alone": "asian",
       "Estimate!!Total!!Native Hawaiian and Other Pacific Islander alone": "pacific",
       "Estimate!!Total!!Some other race alone": "other",
       "Estimate!!Total:": "total",
       "Estimate!!Total:!!White alone": "white",
       "Estimate!!Total:!!Black or African American alone": "black",
       "Estimate!!Total:!!American Indian and Alaska Native alone": "native",
       "Estimate!!Total:!!Asian alone": "asian",
       "Estimate!!Total:!!Native Hawaiian and Other Pacific Islander alone": "pacific",
       "Estimate!!Total:!!Some Other Race alone": "other",
       "Estimate!!Total:!!Some other race alone": "other"
}

if not os.path.exists("./raw/lendingClubExtended.csv"):
       # preprocess lending club data
       lendingClub_df = pd.read_csv("./raw/lendingClub/accepted_2007_to_2018Q4.csv.gz", compression="gzip")
       lendingClub_df["fico"] = (lendingClub_df["fico_range_low"] + lendingClub_df["fico_range_high"]) / 2
       lendingClub_df["issue_d"] = pd.to_datetime(lendingClub_df["issue_d"])
       lendingClub_df["year"] = lendingClub_df["issue_d"].dt.year
       lendingClub_df["month"] = lendingClub_df["issue_d"].dt.month
       lendingClub_df = lendingClub_df.drop(columns=["issue_d"])
       lendingClub_df = lendingClub_df.dropna(subset=["zip_code", "year"])
       lendingClub_df["year"] = lendingClub_df["year"].astype(int)
       lendingClub_df.columns = ["lc-" + x if x not in ["zip_code", "year"] else x for x in lendingClub_df.columns]

       # preprocess census data
       census_df = pd.DataFrame()
       for file in glob("./raw/census/ACSDT5Y*.B02001-Data.csv"):
           temp = pd.read_csv(file, header= 1)
           temp["year"] = int(file.split("ACSDT5Y")[1][:4])
           temp = temp.rename(columns=census_dict)
           temp["zip_code"] = temp["Geographic Area Name"].str.extract(r"(\d{5})").astype(int)[0].apply(lambda x: str(x)[:3] + "xx")
           census_df = pd.concat([census_df, temp.loc[:, ["zip_code", "year"] + list(set(census_dict.values()))]], axis= 0)
       census_df = census_df.groupby(["zip_code", "year"]).sum().reset_index()
       census_df["non-white"] = census_df["total"] - census_df["white"]
       for ethnicity in ["white", "non-white", "black", "native", "asian", "pacific", "other"]:
              census_df["{}_ratio".format(ethnicity)] = census_df[ethnicity] / census_df["total"]
       census_df = census_df.dropna(subset=["zip_code", "year"])
       census_df["year"] = census_df["year"].astype(int)
       census_df.columns = ["cs-" + x if x not in ["zip_code", "year"] else x for x in census_df.columns]

       # preprocess housing data
       housing_df = pd.read_excel("./raw/housing/HPI_AT_3zip.xlsx", dtype= {"Three-Digit ZIP Code": object})
       housing_df = housing_df.rename(columns={"Year": "year", "Three-Digit ZIP Code": "zip_code", "Index (NSA)": "housing_index"})
       housing_df = housing_df.loc[(housing_df["year"] >= 2007) & (housing_df["year"] <= 2018) & (housing_df["Index Type"] == "Native 3-Digit ZIP index"), ["zip_code", "year", "housing_index"]]
       housing_df["zip_code"] = housing_df["zip_code"].apply(lambda x: f'{x:03}' + "xx")
       housing_df = housing_df.groupby(["zip_code", "year"]).mean().reset_index()
       housing_df["year"] = housing_df["year"].astype(int)
       housing_df.columns = ["hg-" + x if x not in ["zip_code", "year"] else x for x in housing_df.columns]

       # preprocess economic data
       economic_df = pd.read_csv("./raw/economy/API_USA_DS2_en_csv_v2_1512.csv", skiprows= 4)
       economic_df = economic_df.drop(columns=['Country Name', 'Country Code', 'Indicator Code', 'Unnamed: 68']).transpose()
       economic_df = economic_df.rename(columns= economic_df.iloc[0])
       economic_df = economic_df.drop(economic_df.index[0])
       economic_df["year"] = economic_df.index.astype(int)
       economic_df = economic_df.reset_index(drop= True)
       economic_df = economic_df.loc[:, ["GDP growth (annual %)", "Unemployment, total (% of total labor force) (national estimate)", "Inflation, consumer prices (annual %)", "year"]]
       economic_df = economic_df.rename(columns={"GDP growth (annual %)": "gdp_growth", "Unemployment, total (% of total labor force) (national estimate)": "unemployment", "Inflation, consumer prices (annual %)": "inflation"})
       economic_df.columns = ["ec-" + x if x not in ["year"] else x for x in economic_df.columns]

       # combine data and save
       df = pd.merge(lendingClub_df, census_df, on= ["zip_code", "year"], how= "inner")
       df = pd.merge(df, housing_df, on= ["zip_code", "year"], how= "left")
       df = pd.merge(df, economic_df, on= "year", how= "left")
       df.to_csv("./raw/lendingClubExtended.csv", index= False)

# preprocess lending club data
df = pd.read_csv("./raw/lendingClubExtended.csv")

# set target variable
df = df.loc[df["lc-loan_status"].isin(["Fully Paid", "Charged Off"]), :]
df["lc-default"] = (df["lc-loan_status"] == "Charged Off").astype(int)

# get only loans from pre-dominantly white and non-white areas
df = df.loc[(df["cs-white_ratio"] > 0.75) | (df["cs-non-white_ratio"] > 0.75), :]
df["ethnicity"] = df.loc[:, ["cs-white_ratio", "cs-non-white_ratio"]].idxmax(axis=1).str.split("_").str[0].str.replace("cs-", "")

## evaluate correlation between ethnicity and additional data
temp = df.loc[:, ["ethnicity", "hg-housing_index", "ec-gdp_growth", "ec-unemployment", "ec-inflation"]]
temp["ethnicity"] = LabelEncoder().fit_transform(temp["ethnicity"])
corr_pd1 = temp.loc[:, ["ethnicity", "ec-gdp_growth", "ec-unemployment", "ec-inflation"]].corr()["ethnicity"][1:].mean()
corr_pd2 = temp.loc[:, ["ethnicity", "hg-housing_index" ]].corr()["ethnicity"][1:].mean()
print("Avg Correlation between Ethnicity and Economic Data: {}".format(corr_pd1))
print("Avg Correlation between Ethnicity and Housing Data: {}".format(corr_pd2))

# assign US region based on US state
df["region"] = df["lc-addr_state"].apply(get_us_region)
df["subregion"] = df["lc-addr_state"].apply(get_us_econ_region)

# create top regions
df["topregion"] = df["region"].apply(lambda x: "West" if x in ["Midwest", "West"] else "East" if x in ["Northeast", "South"] else "Unknown")

# select the most important lc columns
##lc_columns = ["lc-loan_amnt", "lc-term", "lc-int_rate", "lc-emp_title", "lc-emp_length", "lc-home_ownership", "lc-annual_inc", "lc-verification_status", "lc-issue_d", "lc-purpose", "zip_code", "lc-addr_state", "lc-fico_range_low", "lc-fico_range_high", "lc-open_acc", "lc-pub_rec", "lc-initial_list_status", "lc-application_type", "lc-annual_inc_joint", "lc-verification_status_joint", "lc-open_act_il", "lc-pub_rec_bankruptcies", "lc-tax_liens", "lc-fico", "lc-default"]
lc_columns = ["lc-loan_amnt", "lc-term", "lc-int_rate", "lc-emp_length", "lc-home_ownership", "lc-annual_inc", "lc-verification_status", "lc-issue_d", "lc-purpose", "zip_code", "lc-addr_state", "lc-fico_range_low", "lc-fico_range_high", "lc-open_acc", "lc-pub_rec", "lc-initial_list_status", "lc-application_type", "lc-annual_inc_joint", "lc-verification_status_joint", "lc-open_act_il", "lc-pub_rec_bankruptcies", "lc-tax_liens", "lc-fico", "lc-default"]
df = df.drop(columns= [x for x in df.columns if (x not in lc_columns) and (x[:2] == "lc")])

# select the most important cs columns
cs_columns = ["cs-white_ratio", "cs-non-white_ratio"]
df = df.drop(columns= [x for x in df.columns if (x not in cs_columns) and (x[:2] == "cs")])

# select the most important hg columns
hg_columns = ["hg-housing_index"]
df = df.drop(columns= [x for x in df.columns if (x not in hg_columns) and (x[:2] == "hg")])

# set target columns
df = df.rename(columns={"lc-default": "tg-default", "lc-int_rate": "tg-int_rate"})
discr = KBinsDiscretizer(n_bins= 7, encode= "ordinal", strategy= "quantile")
df["tg-int_rate_cat"] = discr.fit_transform(df["tg-int_rate"].values.reshape(-1, 1)).astype(int)
with open("./intRateDiscretizer.pkl", "wb") as f:
       pickle.dump(discr, f)

# preprocess data
df["lc-term"] = df["lc-term"].str.extract(r"(\d+)").astype(int)
df.loc[df["lc-emp_length"] == "< 1 year", "lc-emp_length"] = "0 years"
df.loc[df["lc-emp_length"] == "10+ years", "lc-emp_length"] = "10 years"
df["lc-emp_length"] = df["lc-emp_length"].str.extract(r"(\d+)").astype(float)
df["zip_code"] = df["zip_code"].str.extract(r"(\d{3})").astype(int)
labelEncoderDict = {}
df["lc-verification_status_joint"] = df["lc-verification_status_joint"].fillna("None")
for col in ["lc-home_ownership", "lc-verification_status", "lc-purpose", "lc-addr_state", "lc-initial_list_status", "lc-application_type", "lc-verification_status_joint", "lc-addr_state"]:
       labelEncoderDict[col] = LabelEncoder()
       df[col] = labelEncoderDict[col].fit_transform(df[col])
with open("./labelEncoderDict.pkl", "wb") as f:
       pickle.dump(labelEncoderDict, f)
df = df.fillna(0)

# normalize the data
normalizerDict = {}
for col in ["lc-loan_amnt", "lc-term", "lc-emp_length", "lc-annual_inc", "lc-fico_range_low", "lc-fico_range_high", "lc-open_acc", "lc-annual_inc_joint", "lc-open_act_il", "lc-tax_liens", "lc-fico", "hg-housing_index", "ec-gdp_growth", "ec-unemployment", "ec-inflation"]:
       normalizerDict[col] = StandardScaler()
       df[col] = normalizerDict[col].fit_transform(df[col].values.reshape(-1, 1))
with open("./normalizerDict.pkl", "wb") as f:
       pickle.dump(normalizerDict, f)

## descriptive statistics
def ethnicity_ratio(x):
       return sum(x == "non-white") / x.count()
descr = df.groupby("subregion").agg({"region": "count", "tg-default": np.mean, "tg-int_rate": np.mean, "ethnicity": ethnicity_ratio})
print(descr)
descr.to_excel("./descriptiveStatistics.xlsx")

# save the preprocessed data
df.to_csv("./loanDataPreprocessed.csv", index= False)