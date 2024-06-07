from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from scipy import stats

def apply_blackAndWhite_to_facetgrid(g, plot_type='bar', hatches= ['xx', 'oo', '--', '||', '//', 'OO', '..', '**'], markers=['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', '+']):
    if plot_type == 'bar' and hatches is None:
        raise ValueError("Hatch patterns must be provided for bar plots.")
    if plot_type == 'scatter' and markers is None:
        raise ValueError("Marker styles must be provided for scatter plots.")
    if plot_type == 'bar':
        color_hatch_map = {}
        hatch_index = 0
        for ax in g.axes.flat:
            for patch in ax.patches:
                color = patch.get_facecolor()
                color_key = tuple(color)
                if color_key not in color_hatch_map:
                    current_hatch = hatches[hatch_index % len(hatches)]
                    color_hatch_map[color_key] = current_hatch
                    hatch_index += 1
                patch.set_hatch(color_hatch_map[color_key])
                patch.set_facecolor('none')  # Remove the fill color
        if g._legend:
            for patch in g._legend.legend_handles:
                color = patch.get_facecolor()
                color_key = tuple(color)
                patch.set_hatch(color_hatch_map[color_key])
                patch.set_facecolor('none')
    else:
        raise ValueError("Unsupported plot type. Currently supports 'bar' and 'scatter'.")

perf_df = pd.DataFrame()
for file in glob("./results/sub/30/*.xlsx"):
    sub_perf_df = pd.read_excel(file)
    perf_df = pd.concat((perf_df, sub_perf_df), axis=0)

perf_df = perf_df.reset_index(drop= True)
if "Unnamed: 0" in perf_df.columns:
    perf_df = perf_df.drop(columns= ["Unnamed: 0"])
#perf_df = perf_df.loc[perf_df["cv_i"] < 15]
regions = perf_df["region"].unique()

df = pd.read_csv("./data/loanDataPreprocessed.csv")
defaults = df[["tg-default", "tg-int_rate_cat", "tg-int_rate", "ethnicity"]].groupby("ethnicity").mean()
defaults = defaults.loc["non-white"] - defaults.loc["white"]

# normalize by baseline
perf_df_baseline = perf_df.loc[perf_df["type"] == "baseline"]
perf_df_norm = perf_df.loc[perf_df["type"] != "baseline"]
perf_df_norm = pd.merge(perf_df_norm, perf_df_baseline, on= ["region", "cv_i"], how="left", suffixes=("", "_base"))
pot_score_columns = ["def_accuracy", "def_f1", "def_rocauc", "def_demoPar", "def_equOdds", "int_accuracy", "int_f1", "int_lrl", "int_lrp", "int_demoPar", "int_rmse", "int_r2", "int_normDiffInPred", "def_normDiffInPred", "int_DiffInPred", "def_DiffInPred", "int_DiffInSource", "def_DiffInSource"]
score_columns = [x for x in pot_score_columns if x in perf_df_norm.columns]
for score in score_columns:
    perf_df_norm[score + "_excs"] = perf_df_norm[score] - perf_df_norm[score + "_base"]

perf_df_baseline= perf_df_baseline.drop(columns= "type")
perf_df_baseline.columns = ["baseline_" + x for x in perf_df_baseline.columns]
perf_df_norm = pd.merge(left= perf_df_norm, right= perf_df_baseline, left_on= ["region", "cv_i"], right_on= ["baseline_region", "baseline_cv_i"], how= "left")
perf_df_norm.to_excel("./results/performanceData.xlsx")

# Default Prediction
## performance
g = sns.FacetGrid(perf_df_norm, col="region")
g.map(sns.barplot, "type", "def_f1_excs", linewidth=1, edgecolor="black")
for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda y, p: "{:.0%}".format(y)))
g.fig.tight_layout()
apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
plt.subplots_adjust(top=0.85)
plt.suptitle("Performance Comparison DSS vs Baseline")
plt.savefig("./results/figures/def_performanceComparison.png")
plt.show()

## filter out underperforming algorithms
underperforming_dss= perf_df_norm.groupby("type")["def_f1_excs"].mean()
underperforming_dss = list(underperforming_dss.loc[underperforming_dss < 0].index)
perf_df_norm_filtered = perf_df_norm.loc[~perf_df_norm["type"].isin(underperforming_dss)]

## show fairness
g = sns.FacetGrid(perf_df_norm_filtered, col="region")
g.map(sns.scatterplot, "type", "def_DiffInPred", color= "black")
def plot_axhline(y, **kwargs):
    ym = y.mean()
    plt.axhline(ym, color=kwargs["kwargs"]["linecolor"], linestyle=kwargs["kwargs"]["linestyle"])
    plt.annotate(kwargs["kwargs"]["linelabel"], xy=(1, ym), xycoords=plt.gca().get_yaxis_transform(), ha="left", color=kwargs["kwargs"]["linecolor"])
g = g.map(plot_axhline, "def_DiffInSource", kwargs= {"linecolor": "black", "linestyle": "--", "linelabel": "real"})
#g = g.map(plot_axhline, "baseline_def_DiffInPred", kwargs= {"linecolor": "black", "linestyle": ":", "linelabel": "baseline"})
g = g.map(plot_axhline, "def_DiffInPred", kwargs= {"linecolor": "black", "linestyle": "-", "linelabel": "mean"})
plt.subplots_adjust(top=0.85)
plt.suptitle("Fairness Comparison DSS methods")
plt.savefig("./results/figures/def_fairnessComparison.png")
plt.show()

## test for statistical difference in fairness per dss type vs baseline/
stats_test_dict = {}
types = perf_df_norm_filtered["type"].unique()
#perf_df_norm_filtered = perf_df_norm_filtered.loc[perf_df_norm_filtered["region"] == "Far West"]
for type in types:
    stats_test_dict[type] = {}
    # test difference to sample diff
    #stats_test_dict[type]["real"] = stats.ttest_ind(perf_df_norm.loc[perf_df_norm["type"] == type, "int_DiffInSource"], perf_df_norm.loc[perf_df_norm["type"] == type, "int_DiffInPred"], equal_var=False).pvalue
    stats_test_dict[type]["real"] = stats.mstats.ttest_1samp(perf_df_norm_filtered.loc[perf_df_norm_filtered["type"] == type, "def_DiffInSource"] - perf_df_norm_filtered.loc[perf_df_norm_filtered["type"] == type, "def_DiffInPred"], 0, axis=0, alternative='greater')[1]

    # test difference to baseline diff
    #stats_test_dict[type]["baseline"] = stats.ttest_ind(perf_df_norm.loc[perf_df_norm["type"] == type, "baseline_int_DiffInPred"], perf_df_norm.loc[perf_df_norm["type"] == type, "int_DiffInPred"], equal_var=False).pvalue
    stats_test_dict[type]["baseline"] = stats.mstats.ttest_1samp(perf_df_norm_filtered.loc[perf_df_norm_filtered["type"] == type, "baseline_def_DiffInPred"] - perf_df_norm_filtered.loc[perf_df_norm_filtered["type"] == type, "def_DiffInPred"], 0, axis=0, alternative='greater')[1]

stats_test_df = pd.DataFrame(stats_test_dict)
stats_test_df.to_excel("./results/def_statisticalComparison.xlsx")
print(stats_test_df)



# Interest Rate Regression
## performance
g = sns.FacetGrid(perf_df_norm, col="region")
g.map(sns.barplot, "type", "int_r2_excs", linewidth=1, edgecolor="black")
for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda y, p: "{:.0%}".format(y)))
g.fig.tight_layout()
apply_blackAndWhite_to_facetgrid(g, plot_type='bar')
plt.subplots_adjust(top=0.85)
plt.suptitle("Performance Comparison DSS vs Baseline")
plt.savefig("./results/figures/int_performanceComparison.png")
plt.show()

## filter out underperforming algorithms
underperforming_dss= perf_df_norm.groupby("type")["int_r2_excs"].mean()
underperforming_dss = list(underperforming_dss.loc[underperforming_dss < 0].index)
perf_df_norm_filtered = perf_df_norm.loc[~perf_df_norm["type"].isin(underperforming_dss)]

## show fairness
g = sns.FacetGrid(perf_df_norm_filtered, col="region")
g.map(sns.scatterplot, "type", "int_DiffInPred", color= "black")
def plot_axhline(y, **kwargs):
    ym = y.mean()
    plt.axhline(ym, color=kwargs["kwargs"]["linecolor"], linestyle=kwargs["kwargs"]["linestyle"])
    plt.annotate(kwargs["kwargs"]["linelabel"], xy=(1, ym), xycoords=plt.gca().get_yaxis_transform(), ha="left", color=kwargs["kwargs"]["linecolor"])
g = g.map(plot_axhline, "int_DiffInSource", kwargs= {"linecolor": "black", "linestyle": "--", "linelabel": "real"})
#g = g.map(plot_axhline, "baseline_int_DiffInPred", kwargs= {"linecolor": "black", "linestyle": ":", "linelabel": "baseline"})
g = g.map(plot_axhline, "int_DiffInPred", kwargs= {"linecolor": "black", "linestyle": "-", "linelabel": "mean"})
plt.subplots_adjust(top=0.85)
plt.suptitle("Fairness Comparison DSS methods")
plt.savefig("./results/figures/int_fairnessComparison.png")
plt.show()

## test for statistical difference in fairness per dss type vs baseline/
stats_test_dict = {}
types = perf_df_norm_filtered["type"].unique()
#perf_df_norm_filtered = perf_df_norm_filtered.loc[perf_df_norm_filtered["region"] == "Far West"]
for type in types:
    stats_test_dict[type] = {}
    # test difference to sample diff
    #stats_test_dict[type]["real"] = stats.ttest_ind(perf_df_norm.loc[perf_df_norm["type"] == type, "int_DiffInSource"], perf_df_norm.loc[perf_df_norm["type"] == type, "int_DiffInPred"], equal_var=False).pvalue
    stats_test_dict[type]["real"] = stats.mstats.ttest_1samp(perf_df_norm_filtered.loc[perf_df_norm_filtered["type"] == type, "int_DiffInSource"] - perf_df_norm_filtered.loc[perf_df_norm_filtered["type"] == type, "int_DiffInPred"], 0, axis=0, alternative='greater')[1]

    # test difference to baseline diff
    #stats_test_dict[type]["baseline"] = stats.ttest_ind(perf_df_norm.loc[perf_df_norm["type"] == type, "baseline_int_DiffInPred"], perf_df_norm.loc[perf_df_norm["type"] == type, "int_DiffInPred"], equal_var=False).pvalue
    stats_test_dict[type]["baseline_DSSbetter"] = stats.mstats.ttest_1samp(perf_df_norm_filtered.loc[perf_df_norm_filtered["type"] == type, "baseline_int_DiffInPred"] - perf_df_norm_filtered.loc[perf_df_norm_filtered["type"] == type, "int_DiffInPred"], 0, axis=0, alternative='greater')[1]
    stats_test_dict[type]["baseline_DSSnonequal"] = stats.mstats.ttest_1samp(perf_df_norm_filtered.loc[perf_df_norm_filtered["type"] == type, "baseline_int_DiffInPred"] - perf_df_norm_filtered.loc[perf_df_norm_filtered["type"] == type, "int_DiffInPred"], 0, axis=0, alternative='two-sided')[1]

stats_test_df = pd.DataFrame(stats_test_dict)
stats_test_df.to_excel("./results/int_statisticalComparison.xlsx")
print(stats_test_df)