import hddm
import numpy as np
import matplotlib as plt
from scipy import stats
import pandas as pd

class HDDMmodelMaker():

    def fit_model(self, data, size, burn):
        model = hddm.HDDM(data, bias = True)
        #model.find_starting_values()
        model.sample(size, burn = burn)
        return model


    def get_stats(self, model, subjects = False):
        stats_df = model.gen_stats()
        stats_df.index.name = 'parameter'
        stats_df.reset_index(inplace=True)
        if subjects:
            new = stats_df["parameter"].str.split("_subj.", n = 1, expand = True)
            stats_df["param"]= new[0]
            stats_df["subject"]= new[1]
            stats_df.drop(columns =["parameter"], inplace = True)
        return stats_df

    def plot_params(self, stats_df, title):
        parameters = ["a", "v", "z", "t"]
        fig, axs = plt.subplots(4, 4)
        fig.set_size_inches(13, 13)
        fig.suptitle('Correlation plots with Pearson coefficient, %s, (subjects model)' % (title), fontsize=16)
        i = 0
        j = 0
        for p2 in parameters:
            j = 0
            for p in parameters:
                x = stats_df.loc[stats_df.param == p, "mean"]
                y = stats_df.loc[stats_df.param == p2, "mean"]
                axs[i, j].scatter(x, y)
                axs[i, j].set_title("Corr: %f, p value: %s" % (
                np.round(stats.pearsonr(x, y)[0], 2), np.round(stats.pearsonr(x, y)[1], 4)))
                j = j + 1
            i = i + 1

        axs.flat[0].set(ylabel='a')
        axs.flat[4].set(ylabel='v')
        axs.flat[8].set(ylabel='z')
        axs.flat[12].set(ylabel='t')
        axs.flat[12].set(xlabel='a')
        axs.flat[13].set(xlabel='v')
        axs.flat[14].set(xlabel='z')
        axs.flat[15].set(xlabel='t')


