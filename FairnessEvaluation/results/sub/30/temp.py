import pandas as pd

algorithms = [
    "activeLearning",
    "baseline",
    "dataSharing",
    "federatedLearning",
    "pretrainedModel",
    "publicData"
]

for algorithm in algorithms:
    df_clf = pd.read_excel("./clf/preds/{}.xlsx".format(algorithm))
    df_reg = pd.read_excel("./reg/{}.xlsx".format(algorithm))
    df = pd.merge(left= df_clf, right= df_reg, on= ["region", "type", "cv_i"], how="outer")
    df.to_excel("./{}.xlsx".format(algorithm), index= False)