import pandas as pd, numpy as np
import os

_hanesmap = {
    "ID": "nca0_id",
    #
    "GS_use": "_grwhich",
    "GS_mea_r_1": "__gripr1",
    "GS_mea_r_2": "_gripr2",
    "GS_mea_l_1": "_gripl1",
    "GS_mea_l_2": "_gripl2",
    #
    "BIA_LRA": "_armrm",
    "BIA_LLA": "_armlm",
    "BIA_LRL": "_legrm",
    "BIA_LLL": "_leglm",
    #
    "HE_glu": "_glu0",
    "HE_HbA1c": "_hba1c",
    #
    "HE_ht": "_height",
    "HE_wt": "_weight",
    "HE_wc": "_waist",
    #
    "HE_chol": "_tchl",
    "HE_TG": "_tg",
    "HE_LDL_drct": "_ldl",
    "HE_HDL_st2": "_hdl",
    "HE_ast": "_ast",
    "HE_alt": "_alt",
    #
    "HE_HB": "_hb",
    "HE_HCT": "_hct",
    "HE_WBC": "_wbc",
    "HE_RBC": "_rbc",
    "HE_Bplt": "_plat",
    "HE_Uacid": "_uricacid",
    "HE_hsCRP": "_crp",
    "HE_BUN": "_bun",
    "HE_mPLS": "_pulse",
    "age": "_age",
    "sex": "_sex",
}

d = os.path.join(os.path.dirname(__file__), "data_hanes")
df = os.path.join(os.path.dirname(__file__), "data_fixed_hanes")
years = [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]

for y in years:
    data = pd.read_spss(os.path.join(d, f"HN{y}_ALL.sav"))
    datafixed = pd.DataFrame()
    for x in data.columns:
        if x in _hanesmap:
            datafixed[_hanesmap.get(x, x)] = data[x]

    int_var = ["sex", "age", "grwhich"]
    for x in int_var:
        if f"_{x}" in datafixed.columns:
            datafixed[f"_{x}"] = datafixed[f"_{x}"].fillna(9).astype(int)

    datafixed.to_csv(os.path.join(df, f"data_track_{y}.csv"), index=False)
