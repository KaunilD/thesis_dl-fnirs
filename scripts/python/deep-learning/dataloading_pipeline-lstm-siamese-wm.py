import pandas as pd
from scipy.io import loadmat
import tables
import numpy as np

import glob
import os
import random
from random import shuffle

from sklearn.preprocessing import normalize

"""
Author: Trevor Jordan Grant.
default4: spatial/verbal
"""
# Dictionary is labeled such that task labels in conditions files will have
# more than one multilabeling schema.

# To add more multilabeling schema - include them in the task Dictionary.
# (The sub-dictionary where the task label is the key.)

# default4 = ["VerbalWM", "SpatialWM", "VisualPerceptual", "AuditoryPerceptual"]
# every label in default 4 has discrete values of 'off', 'low', 'high'

# default3 = ["WM", "VisualPerceptual", "AuditoryPerceptual"]
# every label in default 3 has discrete values of 'off', 'low', 'high'

cog_load_label_dict = {
# Mindfulness task labels.
                       "nb": {
                              "default4": ["high", "off", "low", "off"],
                              "default3": ["high", "low", "off"],
                             },
                       "anb": {
                               "default4": ["high", "off", "off", "low"],
                               "default3": ["high", "off", "low"],
                              },
                       "ewm": {
                               "default4": ["low", "off", "high", "off"],
                               "default3": ["low", "high", "off"]
                              },
                        "cr": {
                               "default4": ["off", "off", "off", "off"],
                               "default3": ["off", "off", "off"],
                              },
                        "rt": {
                               "default4": ["off", "off", "low", "off"],
                               "default3": ["off", "low", "off"],
                              },
                        "es": {
                               "default4": ["off", "off", "low", "off"],
                               "default3": ["off", "low", "off"],
                              },
                       "gng": {
                               "default4": ["off", "off", "low", "off"],
                               "default3": ["off", "low", "off"],
                              },
            "adaptive_words": {
                               "default4": ["off", "off", "low", "off"],
                               "default3": ["off", "low", "off"],
                              },
                   "go_nogo": {
                               "default4": ["off", "off", "low", "off"],
                               "default3": ["off", "low", "off"],
                              },
                     "nback": {
                               "default4": ["high", "off", "low", "off"],
                               "default3": ["high", "low", "off"],
                              },
                    "posner": {
                               "default4": ["off", "off", "low", "off"],
                               "default3": ["off", "low", "off"],
                              },
                 "simple_rt": {
                               "default4": ["off", "off", "low", "off"],
                               "default3": ["off", "low", "off"],
                              },
             "visual_search": {
                               "default4": ["off", "off", "high", "off"],
                               "default3": ["off", "high", "off"],
                              },
                      }


def strings_to_vectors(string_labels, as_list=False):
    """Maps strings in dict to interger values.
    Args:
        string_labels(list): The string label value of load.
        as_list(bool): False, if True, return list instead of np.array()
    Returns:
        labels as np.array()
    """

    maps = {
            "off": 0,
            "low": 1,
            "high": 2,
           }

    if as_list:
        return [maps[label] for label in string_labels]
    return np.array([maps[label] for label in string_labels])


def return_label(task, label_type="default3", as_strings=False):
    """Returns a label from the cog_load_label_dict.
    Args:
        task(str): The task label from the coditions file.
        label_type(string): The label schema used for the model.
        as_strings(bool): False, if True, return string (in list) values instead.
    Returns:
        labels(np.array): Under defaults labels will be returned as interger
        values in a np.array().
    """
    if as_strings:
        return cog_load_label_dict[task][label_type]
    return strings_to_vectors(cog_load_label_dict[task][label_type])

channel_52_5x11_mat = {
           1:[0,1],  2:[0,2],  3:[0,3],  4:[0,4],  5:[0,5],  6:[0,6],  7:[0,7],  8:[0,8],  9:[0,9], 10:[0,10],
11:[1,0], 12:[1,1], 13:[1,2], 14:[1,3], 15:[1,4], 16:[1,5], 17:[1,6], 18:[1,7], 19:[1,8], 20:[1,9], 21:[1,10],
          22:[2,1], 23:[2,2], 24:[2,3], 25:[2,4], 26:[2,5], 27:[2,6], 28:[2,7], 29:[2,8], 30:[2,9], 31:[2,10],
32:[3,0], 33:[3,1], 34:[3,2], 35:[3,3], 36:[3,4], 37:[3,5], 38:[3,6], 39:[3,7], 40:[3,8], 41:[3,9], 42:[3,10],
          43:[4,1], 44:[4,2], 45:[4,3], 46:[4,4], 47:[4,5], 48:[4,6], 49:[4,7], 50:[4,8], 51:[4,9], 52:[4,10]
}

def get_52_5x11_mat(data):
    # returns a matrix of size 5x11.
    mat = np.zeros((5, 11))
    for idx, i in enumerate((data)):
        loc = channel_52_5x11_mat[idx+1]
        mat[loc[0], loc[1]] = i
    return mat

channel_52_5x22_mat = {
           1:[0,1],  2:[0,3],  3:[0,5],  4:[0,7],  5:[0,9],  6:[0,11],  7:[0,13],  8:[0,15],  9:[0,17], 10:[0,19],
11:[1,0], 12:[1,2], 13:[1,4], 14:[1,6], 15:[1,8], 16:[1,10], 17:[1,12], 18:[1,14], 19:[1,16], 20:[1,18], 21:[1,20],
          22:[2,1], 23:[2,3], 24:[2,5], 25:[2,7], 26:[2,9], 27:[2,11], 28:[2,13], 29:[2,15], 30:[2,17], 31:[2,19],
32:[3,0], 33:[3,2], 34:[3,4], 35:[3,6], 36:[3,8], 37:[3,10], 38:[3,12], 39:[3,14], 40:[3,16], 41:[3,18], 42:[3,20],
          43:[4,1], 44:[4,3], 45:[4,5], 46:[4,7], 47:[4,9], 48:[4,11], 49:[4,13], 50:[4,15], 51:[4,17], 52:[4,19]
}

def get_52_5x22_mat(data):
    # returns a matrix of size 5x11.
    mat = np.zeros((5, 22))
    for idx, i in enumerate((data)):
        loc = channel_52_5x22_mat[idx+1]
        mat[loc[0], loc[1]] = i
    return mat


def collapse_tasks(tasks, min_dur):
    collapsed_tasks = []
    collapsed_tasks.append(
        tasks[0]
    )
    for i in range(1, len(tasks)):
        t1 = collapsed_tasks[-1]
        t2 = tasks[i]
        if t1["class"] == t2["class"] and (t2["duration"] < min_dur and t1["duration"] < min_dur):

            t1["data"] = np.concatenate((t1["data"], t2["data"]), axis=0)
            t1["duration"]+=t2["duration"]
            t1["end"]=t1["onset"]+t2["duration"]
            # merge and append
        else:
            # just append
            collapsed_tasks.append(t2)
    return collapsed_tasks

def read_tasks(condition, data):
    # conditions, data = csv, mat files
    # tuple containing (class, onset, duration, offset, oxy_data, dxy_data)
    print(condition)
    tasks = []
    # read conditions, data
    c_data = pd.read_csv(condition)
    m_data = loadmat(data)
    # get oxy, dxy data
    oxyDaya = m_data['nirs_data'][0][0][0]
    dxyData = m_data['nirs_data'][0][0][1]
    # iterate through all the tasks here now.
    for idx, key in enumerate(list(c_data.keys())):
        start = 0
        end = 0
        class_ = None
        if 'Task' in key or 'all_benchmarks_fNIRS' in key:
            # get start and end index of the task
            if 'Task' in key:
                start = int(c_data[key][0])
                duration = int(c_data[key][1])
                class_ = c_data[key][2]
            else:
                start = int(c_data[key][2])
                duration = int(c_data[key][3])
                class_ = c_data[key][4]
            if class_ == "adaptive_words" or class_ == "posner" or class_ == "es":
                continue

            end = start + duration

            # visualize heatmap:
            # sns.heatmap(get_52_mat(oxyDaya[0]))

            oxy_series = oxyDaya[start:end, :]
            dxy_series = dxyData[start:end, :]

            # a 100x5x22 list
            oxy_dxy_series_mat = np.zeros((duration,2, 5, 11))

            for ts, (oxy_slice, dxy_slice) in enumerate(zip(oxy_series, dxy_series)):
                oxy_slice = get_52_5x11_mat(oxy_slice)
                dxy_slice = get_52_5x11_mat(dxy_slice)

                #oxy_dxy_series_mat[ts] = np.hstack([oxy_slice, dxy_slice])
                oxy_dxy_series_mat[ts] = np.array([oxy_slice, dxy_slice])
            tasks.append(
                {
                    "class": class_,
                    "onset": start,
                    "end": end,
                    "duration": duration,
                    "data" : oxy_dxy_series_mat
                }
            )
    return tasks

def pad_tasks(tasks):
    lengths = [len(t["data"]) for t in tasks]
    #max_len = max(lengths)
    max_len = 3000
    for t in tasks:
        padded_task = np.zeros(np.concatenate( ([max_len], t["data"].shape[1:]) ))
        padded_task[:min(t["duration"], max_len)] = t["data"][:min(max_len, t["duration"])]
        t["data"] = padded_task
    return tasks


if __name__ == '__main__':

    conditions = sorted(glob.glob('../../../data/multilabel/mats/mindfulness/*.csv'))
    data = sorted(glob.glob('../../../data/multilabel/mats/mindfulness/*.mat'))

    task_data = []
    time_series_length = 10
    """
    default3 labels
    [
        wm,
        v,
        a
    ]
    """

    for idx, (cond, dat) in enumerate(zip(conditions, data)):

        participant_id = os.path.basename(cond)[0:4]

        session_id = os.path.basename(cond) # etc.csv
        session_id = session_id.split(".")[0] # etc
        session_id = session_id[-2:]

        tasks = read_tasks(cond, dat)
        for t in tasks:
            task_data.append(t)
            task_data[-1]["participant_id"] = participant_id
            task_data[-1]["session_id"] = session_id
            task_data[-1]["wl_label"] = return_label(task_data[-1]["class"])
        task_data = collapse_tasks(task_data, min_dur=time_series_length)

    #### GET wm, vl, al (off, low, high) label counts and counts for each type of task

    labels_bin = {"wm":{0:0, 1:0, 2:0}, "vl":{0:0, 1:0, 2:0}, "al":{0:0, 1:0, 2:0}}
    task_cond_bin = {i:{"ts":0, "cnt":0} for i in cog_load_label_dict}
    for t in task_data:
        label = return_label(t["class"])

        task_cond_bin[t["class"]]["cnt"] += 1
        task_cond_bin[t["class"]]["ts"] = t["duration"]

        labels_bin["wm"][label[0]]+=1
        labels_bin["vl"][label[1]]+=1
        labels_bin["al"][label[2]]+=1

    print(task_cond_bin, labels_bin)

    participant_taskdata = {}



    invalids = [
'8210_s2','2003_s1','8208_s2','8203_s1','8204_s1','8212_s1','2014_s2','2006_s2','8208_s2','8206_s2','8203_s1','8213_s1','8204_s1','2004_s1','8201_s1','8215_s1','8219_s1','8211_s1','2001_s1','2006_s1','2006_s1','2006_s2','8210_s1','8210_s2','8210_s2','8218_s1','2017_s1','2019_s1','2003_s1','8209_s1','8208_s1','8208_s1','8208_s2','2012_s1','8220_s1','2007_s1','8205_s2','2011_s1','2011_s2','2015_s2','8204_s2','2013_s1','2013_s1','2013_s2','8212_s2','2002_s2','2004_s1','2004_s1','2014_s1','2014_s1','2014_s2','8215_s1','2001_s1','2006_s1','2006_s2','8210_s2','2017_s1','2019_s1','8209_s2','8208_s2','8208_s2','2012_s1','2012_s2','8216_s2','8216_s2','8206_s2','8206_s2','8220_s1','8220_s1','2007_s1','2007_s1','8203_s1','8203_s1','2011_s1','8213_s1','8213_s1','2015_s1','2015_s1','2015_s2','8204_s1','8204_s2','2013_s1','2013_s2','8212_s1','2002_s1','2002_s2','2004_s1','2004_s1','2014_s1','8201_s1','8201_s1','8219_s1','8219_s1','8211_s1','8211_s2','2001_s1','2001_s1','8221_s2','2006_s1','8210_s1','8210_s1','8210_s2','8218_s1','2017_s1','2019_s1','2019_s1','2003_s1','8209_s1','8209_s1','8209_s2','8208_s1','8208_s1','8208_s2','8208_s2','2012_s1','2012_s1','2012_s2','2012_s2','8216_s1','8216_s1','8216_s2','8216_s2','8206_s1','8220_s1','2007_s1','2007_s1','8205_s2','2015_s1','2013_s2','8219_s1','2006_s2','8210_s2'
    ]


    for t in task_data:
        if t["participant_id"]+'_'+t["session_id"] not in invalids:
            if t["participant_id"] not in participant_taskdata:
                participant_taskdata[t["participant_id"]] = []
            participant_taskdata[t["participant_id"]].append(t)



    participant_ids = list(participant_taskdata.keys())
    print(participant_ids)



    train_ids = participant_ids[:int(0.8*len(participant_ids))]
    val_ids = participant_ids[int(0.8*len(participant_ids)):]
    print(len(train_ids))
    print(len(val_ids))

    TIME_CROP_LENGTH = 300
    # #### Get total rows in wl = wm for each off, low, high
    """
    NORMALIZING PARTICIPANTS
    """
    for participant_id in participant_taskdata:
        durations = []
        data = []

        for task in participant_taskdata[participant_id]:
            data.append(task["data"])
            durations.append(task["duration"])

        cat_tasks = np.concatenate(data)

        mask = cat_tasks == 0
        cat_tasks+=mask*0.0001

        cat_tasks_mean = np.mean(cat_tasks, axis=0)
        cat_tasks_std = np.std(cat_tasks, axis=0)
        cat_tasks-=cat_tasks_mean
        cat_tasks/=cat_tasks_std

        current_ts = 0
        for idx, task in enumerate(participant_taskdata[participant_id]):
            current_task = cat_tasks[current_ts:current_ts+durations[idx]]
            task["data"] = current_task
            current_ts+=durations[idx]

    task_data = pad_tasks(task_data)

    train_labeled_task_bin = {0:[], 1:[], 2:[]}
    for participant_id in train_ids:
        for t in participant_taskdata[participant_id]:

            wm_label = t["wl_label"][0]

            if wm_label in [1, 2]:
                for i in range(0, 60, 20):
                    train_labeled_task_bin[wm_label].append(t["data"][i:i+TIME_CROP_LENGTH])
            else:
                train_labeled_task_bin[0].append(t["data"][:TIME_CROP_LENGTH])
    print( [len(train_labeled_task_bin[0]), len(train_labeled_task_bin[1])])



    val_labeled_task_bin = {0:[], 1:[], 2:[]}
    for participant_id in val_ids:
        for t in participant_taskdata[participant_id]:
            wm_label = t["wl_label"][0]

            if wm_label in [1, 2]:
                for i in range(0, 60, 20):
                    val_labeled_task_bin[wm_label].append(t["data"][i:i+TIME_CROP_LENGTH])
            else:
                val_labeled_task_bin[0].append(t["data"][:TIME_CROP_LENGTH])
    print( [len(val_labeled_task_bin[0]), len(val_labeled_task_bin[1])])

    train_pairs = {0:[], 1:[]}

    # matching pairs
    for i in train_labeled_task_bin:

        lab_tasks_idx = [j for j in range(len(train_labeled_task_bin[i]))]
        lab_tasks_perm = lab_tasks_idx.copy()

        shuffle(lab_tasks_perm)

        while True:
            if not np.any(lab_tasks_idx == lab_tasks_perm):
                break

        for a in lab_tasks_idx:
            for b in lab_tasks_perm:
                if a == b : continue
                train_pairs[0].append(
                    (
                        [train_labeled_task_bin[i][a], i], # i is the wm-GT label
                        [train_labeled_task_bin[i][b], i],
                        0
                    )
                )

    # different pairs
    labels = train_labeled_task_bin.keys()
    label_pairs = [(0, 1), (1, 2), (2, 0)]

    for lab1, lab2 in label_pairs:
        for task1 in train_labeled_task_bin[lab1]:
            for task2 in train_labeled_task_bin[lab2]:
                train_pairs[1].append((
                    [task1, lab1], # lab1, lab2 are the wm-gt labels
                    [task2, lab2],
                    1
                ))

    print(len(train_pairs[0]), len(train_pairs[1]))

    shuffle(train_pairs[0])
    shuffle(train_pairs[1])

    """
        write all data to disk as is
    """

    """
    data_list = []
    for idx, data in enumerate(task_data):
        data_list.append(data)
    np.save("C://Users//dhruv//Development//git//thesis_dl-fnirs//data//multilabel//all//mindfulness\\data", data_list)
    """
    """
    np.save("C://Users//dhruv//Development//git//thesis_dl-fnirs//data//multilabel//all//mindfulness\\data_siamese_train", train_pairs)
    """

    NUM_TRAIN_SAMPLES = 20000
    NUM_TEST_SAMPLES = 10000

    # save matching
    for idx, data in enumerate(train_pairs[0][0:NUM_TRAIN_SAMPLES]):
        print("Saved matching pairs {} of {} to disk.".format(idx+1, NUM_TRAIN_SAMPLES), end='\r')
        np.save("../../../data/multilabel/all/mindfulness/siamese/wm/train/0/" + str(idx), data)
    print()
    # save different
    for idx, data in enumerate(train_pairs[1][0:NUM_TRAIN_SAMPLES]):
        print("Saved different pairs {} of {} to disk".format(idx+1, NUM_TRAIN_SAMPLES), end='\r')
        np.save("../../../data/multilabel/all/mindfulness/siamese/wm/train/1/" + str(idx), data)
    print()
    # ##### validation set
    val_label_examples = {0:[1, 2], 1:[0, 2], 2:[1, 0]}
    val_pairs = []
    for i in val_labeled_task_bin:
        for task in val_labeled_task_bin[i]:

            t2 = val_label_examples[i][0]
            t3 = val_label_examples[i][1]
            val_pairs.append({
                "t1": [task, i],
                "t2": [random.choice(val_labeled_task_bin[t2]), t2],
                "t3": [random.choice(val_labeled_task_bin[t3]), t3],
                "t4": [random.choice(val_labeled_task_bin[i]), i]
            })
    print("Saved validation data to disk.")
    np.save("../../../data/multilabel/all/mindfulness/siamese/wm/validation/data_siamese_val", val_pairs)

    # ##### siamese pairs validation
    siamese_pairs_val = {0:[], 1:[]}
    # matching pairs
    for i in val_labeled_task_bin:

        lab_tasks_idx = [j for j in range(len(val_labeled_task_bin[i]))]
        lab_tasks_perm = lab_tasks_idx.copy()

        shuffle(lab_tasks_perm)

        while True:
            if not np.any(lab_tasks_idx == lab_tasks_perm):
                break

        for a in lab_tasks_idx:
            for b in lab_tasks_perm:
                if a == b : continue
                siamese_pairs_val[0].append(
                    (
                        [val_labeled_task_bin[i][a], i],
                        [val_labeled_task_bin[i][b], i],
                        0
                    )
                )

    # different pairs
    labels = val_labeled_task_bin.keys()
    label_pairs = [(0, 1)]

    for lab1, lab2 in label_pairs:
        for task1 in val_labeled_task_bin[lab1]:
            for task2 in val_labeled_task_bin[lab2]:
                siamese_pairs_val[1].append((
                    [task1, lab1],
                    [task2, lab2],
                    1
                ))



    print(len(siamese_pairs_val[0]), len(siamese_pairs_val[1]))

    shuffle(siamese_pairs_val[0])
    shuffle(siamese_pairs_val[1])

    # save matching
    for idx, data in enumerate(siamese_pairs_val[0][0:NUM_TEST_SAMPLES]):
        print("Saved matching pairs {} of {} to disk".format(idx+1, NUM_TEST_SAMPLES), end='\r')
        np.save("../../../data/multilabel/all/mindfulness/siamese/wm/test/0/" + str(idx), data)
    print()

    # save different
    for idx, data in enumerate(siamese_pairs_val[1][0:NUM_TEST_SAMPLES]):
        print("Saved different pairs {} of {} to disk".format(idx+1, NUM_TEST_SAMPLES), end='\r')
        np.save("../../../data/multilabel/all/mindfulness/siamese/wm/test/1/" + str(idx), data)
    print()
