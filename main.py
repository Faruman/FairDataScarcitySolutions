from glob import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame()
for file in glob("./results/sub/*.xlsx"):
    sub_df = pd.read_excel(file)
    df = pd.concat((df, sub_df), axis=0)

# normalize by baseline
df_baseline = df.loc[df["type"] == "baseline"]
df_results = df.loc[df["type"] != "baseline"]
df_results = pd.merge(df_results, df_baseline, on= ["region", "cv_i"], how="left", suffixes=("", "_base"))
for score in ["def_accuracy", "def_f1", "def_demoPar", "def_equOdds", "int_accuracy", "int_f1", "int_lrl", "int_lrp", "int_demoPar", "int_rmse", "int_r2", "int_normDiffInPred"]:
    df_results[score + "_excs"] = df_results[score] - df_results[score + "_base"]

# default prediction
## performance
#sns.barplot(data=df_results, x="type", y="def_accuracy_excs", ci=None)
#plt.show()

#sns.barplot(data=df_results, x="type", y="def_f1_excs", ci=None)
#plt.show()

## fairness
#sns.barplot(data=df_results, x="type", y="def_demoPar_excs", ci=None)
#plt.show()

#sns.barplot(data=df_results, x="type", y="def_equOdds_excs", ci=None)
#plt.show()

# interest rate prediction
# Classification
## performance
#sns.barplot(data=df_results, x="type", y="int_accuracy_excs", ci=None)
#plt.show()

#sns.barplot(data=df_results, x="type", y="int_f1_excs", ci=None)
#plt.show()

#sns.barplot(data=df_results, x="type", y="int_lrl_excs", ci=None)
#plt.show()

#sns.barplot(data=df_results, x="type", y="int_lrp_excs", ci=None)
#plt.show()

#sns.barplot(data=df_results, x="type", y="int_demoPar_excs", ci=None)
#plt.show()

# Regression
## performance
sns.barplot(data=df_results, x="type", y="int_rmse_excs", ci=None)
plt.show()

sns.barplot(data=df_results, x="type", y="int_r2_excs", ci=None)
plt.show()

## fairness
sns.barplot(data=df_results, x="type", y="int_normDiffInPred_excs", ci=None)
plt.show()
