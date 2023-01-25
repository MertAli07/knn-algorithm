import pandas as pd
import numpy as np
import random
import copy
from math import sqrt
from sklearn.metrics import confusion_matrix

def shuffle_data(db):
    for i in range(len(db) - 1, 0, -1):
        j = random.randint(0, len(db) - 1)
        db[[i,j]] = db[[j, i]]

def euclid(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

def get_neigh(train, test_row, k):
    distances = list()
    for train_row in train:
        dist = euclid(train_row, test_row)
        distances.append((train_row, dist))
    distances.sort(key= lambda tup: tup[1])
    neighbors = list()
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

def predict_class(train, test_row, k):
    neighbors = get_neigh(train, test_row, k)
    output = [row[-1] for row in neighbors]
    prediction = max(set(output), key=output.count)
    return prediction

def get_weighted_neigh(train, test_row, k):
    distances = list()
    for train_row in train:
        dist = euclid(train_row, test_row)
        distances.append((train_row, dist))
    distances.sort(key= lambda tup: tup[1])
    neighbors = list()
    for i in range(k):
        neighbors.append((distances[i][0], distances[i][1]))
    return neighbors

def predict_weighted_class(train, test_row, k):
    neighbors = get_weighted_neigh(train, test_row, k)
    output = dict()
    for i in neighbors:
        if(i[0][-1] in output.keys()):
            output[i[0][-1]] += (1/(i[1] + 0.001))
        else:
            output[i[0][-1]] = (1/(i[1] + 0.001))
    prediction = max(output, key=output.get)
    return prediction

def accuracy_score(db, r):
    return round(sum(db.diagonal())/np.sum(db), r)

def precision_avg(db, r):
    prec = list()
    tp_index = 0
    for row in db:
        prec.append(round(row[tp_index]/np.sum(row), r))
        tp_index += 1
    return round(sum(prec)/len(prec), r)

def recall_avg(db, r):
    rec = list()
    tp_index = 0
    for column in db.T:
        rec.append(round(column[tp_index]/np.sum(column), r))
        tp_index += 1
    return round(sum(rec)/len(rec), r)

def feat_normal(data, r):
    data_temp = data[:, :r]
    num = 0
    for column in data_temp.T:
        col_min = np.min(column)
        col_max = np.max(column)
        for i in range(len(column)):
            data[i,num] = (data[i,num]-col_min)/(col_max-col_min)
        num += 1
    return data

def avg_df(dictionary, mux):
    d = dict()
    d["average of folds"] = []
    first_key = list(dictionary.keys())[0]
    for i in range(len(dictionary[first_key])):
        sum = 0
        for key in dictionary.keys():
            sum += dictionary[key][i]
        d["average of folds"].append(round(sum/5, 2))
    df_end = pd.DataFrame.from_dict(d, orient='index', columns = mux)
    return df_end

def mae(truth, preds):
    sum = 0
    for i in range(len(truth)):
        sum += abs(truth[i] - preds[i])
    return sum/len(truth)


if __name__ == '__main__':
    # PART 1
    # read the input file
    df = pd.read_csv("subset_16P.csv", encoding='cp1254')
    data = df.to_numpy()  # convert it to numpy array
    data = np.delete(data, (0), axis=1)  # delete the IDs

    # change the result from string to int
    my_dict = {"ESTJ": 0, "ENTJ": 1, "ESFJ": 2, "ENFJ": 3, "ISTJ": 4, "ISFJ": 5, "INTJ": 6, "INFJ": 7,
               "ESTP": 8, "ESFP": 9, "ENTP": 10, "ENFP": 11, "ISTP": 12, "ISFP": 13, "INTP": 14, "INFP": 15}
    for i in range(len(data[:,60])):
        data[i, 60] = my_dict.get(data[i, 60])

    # shuffle data
    shuffle_data(data)

    results = dict()
    results["fold0"] = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    results["fold1"] = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    results["fold2"] = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    results["fold3"] = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    results["fold4"] = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    accs = copy.deepcopy(results)
    accs_keys = list(accs)

    precs = copy.deepcopy(results)
    precs_keys = list(precs)

    recs = copy.deepcopy(results)
    recs_keys = list(recs)

    for selection in [0, 1]:
        if(selection == 0): # without feature normalization
            # apply 5-fold cross validation
            folded = np.array_split(data, 5)
        else: # with feature normalization
            data = feat_normal(data, 60)
            folded = np.array_split(data, 5)

        fold_num = 0
        for fold in folded:
            subfolded = np.array_split(fold, 5)
            test = subfolded[fold_num]
            train = subfolded[4-fold_num]
            if fold_num == 2:
                train = subfolded[0]
            for i in range(len(subfolded)):
                if fold_num == 2 and i == 0:
                    continue
                if i != fold_num and i != 4-fold_num:
                    np.concatenate((subfolded[i], train))

            fold_num += 1
            # name each divided subset of data as fold_0, fold_1, ..., fold_4 respectively.
            test_x = test[:, :60]
            test_y = test[:, 60]

            train_x = train[:, :60]
            train_y = train[:, 60]
            train_y = train_y.astype("int")

            for k in [1, 3, 5, 7, 9]:
                preds = list()
                preds_w = list()
                for i in range(len(test_y)):
                    preds.append(predict_class(train, test_x[i], k))
                cm = confusion_matrix(test_y.tolist(), preds)
                accs[accs_keys[fold_num-1]][int((k-1)/2)][selection] = accuracy_score(cm, 2)
                precs[precs_keys[fold_num-1]][int((k-1)/2)][selection] = precision_avg(cm, 2)
                recs[recs_keys[fold_num-1]][int((k-1)/2)][selection] = recall_avg(cm, 2)

                for i in range(len(test_y)):
                    preds_w.append(predict_weighted_class(train, test_x[i], k))
                cmw = confusion_matrix(test_y.tolist(), preds_w)
                accs[accs_keys[fold_num-1]][int((k-1)/2)][selection+2] = accuracy_score(cmw, 2)
                precs[precs_keys[fold_num-1]][int((k-1)/2)][selection+2] = precision_avg(cmw, 2)
                recs[recs_keys[fold_num-1]][int((k-1)/2)][selection+2] = recall_avg(cmw, 2)

    mux = pd.MultiIndex.from_product([['k=1', 'k=3', 'k=5', 'k=7', 'k=9'], ['NN k-NN', 'N k-NN', 'NN w-kNN', 'N w-kNN']])
    # accuracy dataframe
    dict_acc = {"fold0":[],"fold1":[],"fold2":[],"fold3":[],"fold4":[]}
    dict_prec = copy.deepcopy(dict_acc)
    dict_rec = copy.deepcopy(dict_acc)
    for i in accs:
        for j in accs.get(i):
            dict_acc.get(i).extend(j)
    df_res = pd.DataFrame.from_dict(dict_acc, orient='index', columns = mux)

    # precision dataframe
    for i in precs:
        for j in precs.get(i):
            dict_prec.get(i).extend(j)
    df_prec = pd.DataFrame.from_dict(dict_prec, orient='index', columns = mux)

    #recall dataframe
    for i in recs:
        for j in recs.get(i):
            dict_rec.get(i).extend(j)
    df_rec = pd.DataFrame.from_dict(dict_rec, orient='index', columns = mux)

    print(df_res)
    print(avg_df(dict_acc, mux))

    print(df_prec)
    print(avg_df(dict_prec, mux))

    print(df_rec)
    print(avg_df(dict_rec, mux))
    # END OF PART1

    # PART 2
    df2 = pd.read_csv("energy_efficiency_data.csv", encoding='cp1254')
    data2 = df2.to_numpy() # convert it to numpy array

    # shuffle data
    shuffle_data(data2)

    mae_heat = copy.deepcopy(results)
    mae_heat_keys = list(mae_heat)

    mae_cool = copy.deepcopy(results)
    mae_cool_keys = list(mae_cool)

    for selection in [0, 1]:
        if(selection == 0): # without feature normalization
            # apply 5-fold cross validation
            folded = np.array_split(data2, 5)
        else: # with feature normalization
            data2 = feat_normal(data2, 8)
            folded = np.array_split(data2, 5)

        fold_num = 0
        for fold in folded:
            subfolded = np.array_split(fold, 5)
            test = subfolded[fold_num]
            train = subfolded[4-fold_num]
            if fold_num == 2:
                train = subfolded[0]
            for i in range(len(subfolded)):
                if fold_num == 2 and i == 0:
                    continue
                if i != fold_num and i != 4-fold_num:
                    np.concatenate((subfolded[i], train))

            fold_num += 1

            # name each divided subset of data as fold_0, fold_1, ..., fold_4 respectively.
            test_x = test[:, :8]
            test_heat_y = test[:, 8]
            test_cool_y = test[:, 9]

            train_heat = train[:, :9]
            train_cool = np.delete(train, 8, 1)

            for k in [1, 3, 5, 7, 9]:
                preds = list()
                preds_w = list()
                for i in range(len(test_heat_y)):
                    preds.append(predict_class(train_heat, test_x[i], k))
                mae_heat[mae_heat_keys[fold_num-1]][int((k-1)/2)][selection] = mae(test_heat_y.tolist(), preds)

                for i in range(len(test_heat_y)):
                    preds_w.append(predict_weighted_class(train_heat, test_x[i], k))
                mae_heat[mae_heat_keys[fold_num-1]][int((k-1)/2)][selection+2] = mae(test_heat_y.tolist(), preds_w)

            for k in [1, 3, 5, 7, 9]:
                preds = list()
                preds_w = list()
                for i in range(len(test_cool_y)):
                    preds.append(predict_class(train_cool, test_x[i], k))
                mae_cool[mae_cool_keys[fold_num-1]][int((k-1)/2)][selection] = mae(test_cool_y.tolist(), preds)
                for i in range(len(test_cool_y)):
                    preds_w.append(predict_weighted_class(train_cool, test_x[i], k))
                mae_cool[mae_cool_keys[fold_num-1]][int((k-1)/2)][selection+2] = mae(test_cool_y.tolist(), preds_w)

    dict_heat = {"fold0":[],"fold1":[],"fold2":[],"fold3":[],"fold4":[]}
    dict_cool = {"fold0":[],"fold1":[],"fold2":[],"fold3":[],"fold4":[]}

    for i in mae_heat:
        for j in mae_heat.get(i):
            dict_heat.get(i).extend(j)
    df_heat = pd.DataFrame.from_dict(dict_heat, orient='index', columns = mux)
    df_heat_avg = avg_df(dict_heat,mux)

    for i in mae_cool:
        for j in mae_cool.get(i):
            dict_cool.get(i).extend(j)
    df_cool = pd.DataFrame.from_dict(dict_cool, orient='index', columns = mux)
    df_cool_avg = avg_df(dict_cool, mux)

    print(df_heat)
    print(df_heat_avg)
    print(df_cool)
    print(df_cool_avg)
