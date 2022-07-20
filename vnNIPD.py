from __future__ import division
import os
import pandas as pd
import HDDMmodelMaker as ddm
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import scipy.stats as st
import hddm
import pandas as pd
import numpy as np

import scipy.stats as st
from matplotlib.ticker import StrMethodFormatter
import math
from kabuki.analyze import check_geweke

colors = {"info":"#004E98", "no info": "#EA0F0B"}
markers = {"info": "d", "no info": "s"}

#change some names from the raw data columns
def change_column_names(data):
    data = data.rename(columns = {"Round": "round", "Move": "action", "Score":"payoff","Time_js[ms]": "rt"})
    del data["Time_php[ms]"]
    return data

#Count cooperative actions
def count_cooperation(r, neighbors, history, user):
    count = 0
    for n in neighbors[user]:
        data = history[n]
        #display(data.loc[data["round"] == r, "action"].item())
        if (data.loc[data["round"] == r - 1, "action"].item() == "C"):
            count = count + 1
    return count

#Format some fields for them to be used by the DDM framework
def format_fiels(data):
    data["rt"] = data["rt"] / 1000
    data["treatment"] = "info"
    data.loc[data.session.isin(['nets4m10', 'nets5m10', 'nets6m10', 'nets7m11']), "treatment"] = "no info"
    data.groupby(["treatment", "session"]).user_id.count()
    data["response"] = 0
    data.loc[data.action == "C", "response"] = 1
    data = hddm.utils.flip_errors(data)
    return data

#Format pairwise data to include the context (number of people cooperating in the previous round)
def get_context(data):
    contexts = []
    for a in data.actions:
        c = 0
        prev = str(a)[1:2]
        if prev == "C":
            c += 1
        contexts.append(c)
    data["context"] = contexts
    return data


def merge_raw_data():
    history = {}
    neighbors = {}
    for filename in os.listdir("./data/vnNIPD/"):
        if filename.startswith("nets"):
            for f2 in os.listdir("./data/vnNIPD/%s/software/data/" % filename):
                if f2.startswith("usuario"):
                    if f2.endswith("history"):
                        actions = pd.read_csv("./data/vnNIPD/%s/software/data/%s" % (filename, f2), sep=" ")
                        actions = change_column_names(actions)
                        actions["session"] = filename
                        user_id = "%s_%s" % (f2.split("history")[0], filename)
                        actions["user_id"] = user_id
                        history[user_id] = actions
                    if f2.startswith("usuario") and len(f2) <= 9:
                        #partners = pd.read_csv("./data/%s/software/data/%s" % (filename, f2), sep=" ")
                        with open("./data/vnNIPD/%s/software/data/%s" % (filename, f2)) as f:
                            partners = f.readlines()
                            partners = [x.strip() for x in partners]
                            partners = ["%s_%s" % (x, filename) for x in partners]
                            neighbors["%s_%s" % (f2, filename)] = partners
    for key, value in history.items():
        actions = value
        actions["context"] = 0
        for r in actions["round"]:
            if r > 1:
                context = count_cooperation(r, neighbors, history, key)
                actions.loc[actions["round"] == r, "context"] = context
                history[key] = actions
    m_data = pd.concat(history.values())
    m_data = format_fiels(m_data)
    return m_data

def normalize(R, T, S, P):
    j = np.max(np.abs([R, T, S, P])) + 1
    R += j
    S += j
    T += j
    P += j
    T_ = T
    R /= T_
    S /= T_
    T /= T_
    P /= T_
    return [R, T, S, P]

def get_RA(data, N, R, T, S, P):
    #"'Computes the Relative Allocation measure'
    #'data: the raw experimental data
    #'N: neighbourhood size, for Von Neumann = 4, Moore = 6, Pairwise = 1'
    #'R, S, T, P are the parameters of the prisoners dilemma'
    #'returns the dataframe with the RA'"
    R, T, S, P = normalize(R, T, S, P)
    data["self"] = 0.0
    data["other"] = 0.0
    for u in data["user_id"].unique():
        f_data = data.loc[(data.user_id == u) & (data["round"] <= 20)]
        for r in sorted(f_data["round"].unique()):
            self = 0.0
            other = 0.0
            context = f_data.loc[f_data["round"] == r, "context"].item()
            if (f_data.loc[f_data["round"] == r, "response"].item() == 1):
                self = (R * context) + (S * (N - context))
                other = (R * context) + (T * (N - context))
            else:
                self = (T * context) + (P * (N - context))
                other = (S * context) + (P * (N - context))
            data.loc[(data.user_id == u) & (data["round"] == r), "self"] = self
            data.loc[(data.user_id == u) & (data["round"] == r), "other"] = other
    return data

def fit_ddm_params(data):
    ddm_params = pd.DataFrame()
    ddm_params["user_id"] = data.user_id.drop_duplicates()
    ddm_params["a"] = 0
    ddm_params["v"] = 0
    ddm_params["t"] = 0
    ddm_params["z"] = 0
    ddm_params["a_sd"] = 0
    ddm_params["v_sd"] = 0
    ddm_params["t_sd"] = 0
    ddm_params["z_sd"] = 0
    modeler = ddm.HDDMmodelMaker()
    for u in ddm_params.user_id:
        model = modeler.fit_model(data.loc[(data["user_id"] == u)], 10000, 200)
        print check_geweke(model, assert_ = False)
        ddm_params.loc[ddm_params.user_id == u, ["a_sd", "v_sd", "t_sd", "z_sd"]] = model.gen_stats()["std"].values
        ddm_params.loc[ddm_params.user_id == u, ["a", "v", "t", "z"]] = model.gen_stats()["mean"].values
    return ddm_params

#p_data = pd.read_csv("./data/PIPD/PIPD.csv")

#p_data = get_context(p_data)
#p_data.to_csv("./data/PIPD/PIPD.csv")

#ddm_pair_fix = fit_ddm_params(p_data.loc[p_data.treatment == "fix"])
#ddm_pair_fix.to_csv("./data/PIPD/ddm_params_fix.csv")
#ddm_pair_ch = fit_ddm_params(p_data.loc[p_data.treatment == "changing"])
#ddm_pair_ch.to_csv("./data/PIPD/ddm_params_ch.csv")

#n_data1 = pd.read_csv("./data/mNIPD/data1.csv")
#n_data2 = pd.read_csv("./data/mNIPD/data2.csv")
#n_dataC = pd.read_csv("./data/mNIPD/dataC.csv")

#ddm_pair_1 = fit_ddm_params(n_data1)
#ddm_pair_1.to_csv("./data/mNIPD/ddm_params_1.csv")
#ddm_pair_2 = fit_ddm_params(n_data2)
#ddm_pair_2.to_csv("./data/mNIPD/ddm_params_2.csv")
#ddm_pair_C = fit_ddm_params(n_dataC)
#ddm_pair_C.to_csv("./data/mNIPD/ddm_params_C.csv")
