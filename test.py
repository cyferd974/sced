import pandas as pd
import numpy as np
from sced.utils import SimulateMultipleBaselineDatasets, DualCriterionSigPanels

n_repeats = 100
n_points = 18
n_patients = 6

df = SimulateMultipleBaselineDatasets(n_repeats, n_patients, n_points, effect_size=2.0, ar=0.2, min_baseline=5, min_phase=5)
typeI_error, power, df_sig = DualCriterionSigPanels(df, alpha=0.05)

df1 = df.loc[1]
AB, phases = df1.values[0, :n_points], df1.values[0, n_points:]
df_test = pd.DataFrame(np.hstack([AB.reshape((-1, 1)), phases.reshape((-1, 1)).astype(int)]), columns=["score", "phases"])
df_test.loc[df_test.phases == 0.0, "phases"] = "A"
df_test.loc[df_test.phases == 1.0, "phases"] = "B"
df_test.to_csv("./data/test_sced.txt", sep="\t", index=False)

df.loc[0].iloc[:6].to_csv("d:/test.csv", sep=";")