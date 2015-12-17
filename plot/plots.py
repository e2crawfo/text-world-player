''' plot R, Q, completion rate for multiple files at the same time'''

import os
import matplotlib.pyplot as plt
import seaborn
seaborn.set(style="white")
import palettable

import numpy as np
import pandas as pd

COLORS = [
    seaborn.xkcd_rgb[c] for c in
    ["cornflower blue", "amber", "faded green", "dusty purple",
     "pale red", "salmon", "denim blue",
     "dark sea green"]]

COLORS =[
    (0.89411765336990356, 0.10196078568696976, 0.1098039224743836),
    (0.21602460800432688, 0.49487120380588578, 0.71987698697576341),
    (0.30426760128900115, 0.68329106055054012, 0.29293349969620797),
    (0.60083047361934894, 0.30814303335021531, 0.63169552298153153),
    (1.0, 0.50591311045721454, 0.0031372549487094226),
    (0.99315647868549106, 0.98700499826786559, 0.19915417450315831),
    (0.65845446095747096, 0.34122261685483596, 0.17079585352364723),
    (0.95850826852461857, 0.50846600392285535, 0.7449288887136124),
    (0.60000002384185791, 0.60000002384185791, 0.60000002384185791)]



def plot(
        x, y, labels, xlabel, ylabel, title,
        ylim=(0, 1.1), xbuffer=0.005,
        linestyle=None, linemap=None, color_cycle=None):
    """ Each row of `y' is a different line.  """

    if color_cycle:
        plt.gca().set_color_cycle(color_cycle)

    xticks = set()
    for i, (d, l) in enumerate(zip(y, labels)):
        ls = "-" if linestyle is None else linestyle[i]
        #idx = i if linemap is None else linemap[i]

        xticks |= set(x)
        plt.plot(x, d, label=l, linestyle=ls)  # color=COLORS[idx],

    plt.xlim((min(xticks)-xbuffer, max(xticks)+xbuffer))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(ylim)

directory = "/home/eric/snorlax/data/comp599_project"
pure_directory = os.path.join(directory, "good")
transfer_directory = os.path.join(directory, "good_transfer")
plot_directory = "/home/eric/comp599/project/plots/"

xlabel = "# Epochs"
quest_ylabel = "# Quests"
reward_ylabel = "Reward"
quest_title = "Avg Quests Compl. per Ep"
reward_title = "Avg Reward per Ep"

quest_ylim = (0.0, 1.1)
reward_ylim = (-2.0, 1.1)


def make_plot(save, labels, directories, big=False, color_cycle=None):
    if big:
        fig = plt.figure(figsize=(10, 5))
    else:
        fig = plt.figure(figsize=(5, 3))

    linestyle = ["-" if "Deep" else "-." for l in labels]
    linemap = None

    # Quest Plot
    fig.add_subplot(1, 2, 1)

    quest_data = [
        pd.read_csv(os.path.join(d, 'test_quest1.log')) for d in directories]
    min_length = min(len(q) for q in quest_data)
    quest_data = [q[:min_length] for q in quest_data]

    x = np.arange(min_length)

    plot(
        x, quest_data, labels, xlabel, quest_ylabel,
        quest_title, ylim=quest_ylim,
        linestyle=linestyle, linemap=linemap, color_cycle=color_cycle)
    plt.axhline(1.0, color='k', linestyle='dashed', linewidth=1.0)

    # Reward Plot
    fig.add_subplot(1, 2, 2)

    reward_data = [
        pd.read_csv(os.path.join(d, 'test_avgR.log')) for d in directories]
    min_length = min(len(r) for r in reward_data)
    reward_data = [r[:min_length] for r in reward_data]

    x = np.arange(min_length)

    plot(
        x, reward_data, labels, xlabel, reward_ylabel,
        reward_title, ylim=reward_ylim,
        linestyle=linestyle, linemap=linemap, color_cycle=color_cycle)
    plt.axhline(1.0, color='k', linestyle='dashed', linewidth=1.0)

    if big:
        plt.legend(
            loc='center left', bbox_to_anchor=(1.05, 0.5), prop={'size': 10},
            handlelength=2.0, handletextpad=.5, shadow=False, frameon=False)
        fig.subplots_adjust(wspace=0.3, left=0.1, right=0.75, bottom=0.15)
    else:
        plt.legend(loc=4, shadow=False, prop={'size': 7}, handlelength=0.7)
        fig.subplots_adjust(wspace=0.4, right=0.90, bottom=0.15)

    #plt.show()

    save = os.path.join(plot_directory, save)
    plt.savefig(save)

#plt.gca().set_color_cycle(palettable.colorbrewer.qualitative.Paired_9.mpl_colors)
tableau = palettable.tableau.TableauMedium_10.mpl_colors

# Random plots.
dirs = [
    os.path.join(pure_directory, d)
    for d in ["lookup_lookup_shallow",
              "random_d10_deep_random_deep",
              "random_d10_shallow_random_shallow",
              "lstm_shallow"]]

make_plot(
    "random.pdf", ["Lookup", "Random Deep", "Random Shallow", "LSTM Shallow"], dirs, color_cycle=tableau)

# Sentence plots.
dirs = [
    os.path.join(pure_directory, d)
    for d in [
        "sentence_d10_MEAN_sentence_deep",
        "sentence_d10_MEAN_sentence_shallow",
        "sentence_d10_PROD_sentence_deep",
        "sentence_d10_PROD_sentence_shallow"]]

make_plot(
    "random_sentence.pdf",
    ["Deep, MEAN", "Shallow, MEAN", "Deep, PROD", "Shallow, PROD"],
    dirs, color_cycle=tableau)

# Pure plots.
dirs = [
    os.path.join(pure_directory, d)
    for d in [
        "bob_deep_bob_deep",
        "bob_shallow_bob_shallow",
        "bow_deep_bow_deep",
        "bow_shallow_bow_shallow",
        "mvrnn_d10_MEAN_mvrnn_deep",
        "mvrrn_d10_MEAN_mvrnn_shallow",
        "mvrnn_d10_PROD_mvrnn_deep",
        "mvrrn_d10_PROD_mvrnn_shallow",
        "lstm_deep",
        "lstm_shallow"]]

make_plot(
    "pure.pdf",
    [
        "BoB Deep",
        "BoB Shallow",
        "BoW Deep",
        "BoW Shallow",
        "MV-RNN Deep, MEAN",
        "MV-RNN Shallow, MEAN",
        "MV-RNN Deep, PROD",
        "MV-RNN Shallow, PROD",
        "LSTM Deep",
        "LSTM Shallow"],
    dirs,
    big=True,
    color_cycle=tableau)

# Transfer plots.
dirs = [
    os.path.join(pure_directory, d)
    for d in [
        "mvrnn_d10_MEAN_mvrnn_deep",
        "mvrnn_d10_PROD_mvrnn_deep",
        "mvrrn_d10_PROD_mvrnn_shallow"]]

transfer_dirs = [
    os.path.join(transfer_directory, d)
    for d in [
        "transfer_mvrnn_d10_MEAN_mvrnn_deep",
        "transfer_mvrnn_d10_PROD_mvrnn_deep",
        "transfer_mvrrn_d10_PROD_mvrnn_shallow"]]

dirs = [v for pair in zip(dirs, transfer_dirs) for v in pair]

color_cycle = [
    (1,102,94),
    (128,205,193),
    (140,81,10),
    (191,129,45),
    (33,102,172),
    (67,147,195),]

f = lambda x: (x[0]/255.0, x[1]/255.0, x[2]/255.0)
color_cycle = [f(x) for x in color_cycle]

make_plot(
    "transfer.pdf",
    [
        "Deep, MEAN",
        "Deep, MEAN, Transfer",
        "Deep, PROD",
        "Deep, PROD, Transfer",
        "Shallow, PROD",
        "Shallow, PROD, Transfer"],
    dirs,
    big=True,
    color_cycle=color_cycle)
