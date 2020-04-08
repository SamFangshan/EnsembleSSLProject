import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATASETS = ['satimage','glass','hepatitis', 'magic', 'bupa', 'wisconsin', 'lymphography', 'phoneme', 'titanic', 'abalone', 'spambase', 'sonar', 'wine', 'vowel', 'australian', 'vehicle', 'spectfheart', 'contraceptive', 'twonorm', 'breast', 'crx', 'ecoli', 'saheart', 'automobile', 'segment', 'penbased', 'flare', 'movement_libras', 'banana', 'coil2000', 'tic-tac-toe', 'ring', 'dermatology', 'page-blocks', 'thyroid', 'mammographic', 'iris', 'german', 'cleveland', 'haberman', 'monk-2', 'nursery', 'appendicitis', 'zoo', 'pima', 'chess', 'splice', 'mushroom', 'heart', 'tae', 'led7digit', 'yeast', 'marketing', 'housevotes', 'texture']

DATASETS_CATEGORICAL = ['lymphography', 'abalone', 'sonar', 'breast', 'crx', 'saheart', 'automobile', 'flare', 'tic-tac-toe', 'german', 'nursery', 'chess', 'splice', 'mushroom', 'housevotes']

PERCENTAGES = [10, 20, 30, 40]

PARTITIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

TYPES_OF_DATA = ['tra', 'trs', 'tst']

TYPES = ["tra-l", "tst", "tra-u"]

def convert_to_csv(dataset, percentage, partition, type_of_data):
    dat_filename = 'ssl_{}/{}-ssl{}/{}-ssl{}-10-{}{}.dat'.format(percentage, dataset, percentage, dataset, percentage, partition, type_of_data)
    csv_filename = 'ssl_{}/{}-ssl{}/{}-ssl{}-10-{}{}.csv'.format(percentage, dataset, percentage, dataset, percentage, partition, type_of_data)
    
    print("Converting {}".format(dat_filename))
    f = open(dat_filename, 'r')
    lines = f.readlines()
    f.close()
    attributes = []
    line_num = 0
    for l in lines:
        if l[0] != '@':
            break
        tokens = l.split(' ')
        if tokens[0] == '@attribute':
            token = tokens[1].strip()
            token = token[0: token.find("{")]
            attributes.append(token)
        line_num += 1
    for i in range(line_num, len(lines)):
        lines[i] = lines[i].replace(" ", "")
    new_lines = [','.join(attributes) + '\n']
    new_lines.extend(lines[line_num:])
    f = open(csv_filename, 'w+')
    for l in new_lines:
        f.write(l)
    f.close()
    print("Saved file {}".format(csv_filename))

def separate_labeled_unlabeled(dataset, percentage, partition):
    tra_filename = 'ssl_{}/{}-ssl{}/{}-ssl{}-10-{}{}.csv'.format(percentage, dataset, percentage, dataset, percentage, partition, "tra")
    trs_filename = 'ssl_{}/{}-ssl{}/{}-ssl{}-10-{}{}.csv'.format(percentage, dataset, percentage, dataset, percentage, partition, "trs")
    tra_filename_labeled = 'ssl_{}/{}-ssl{}/{}-ssl{}-10-{}{}.csv'.format(percentage, dataset, percentage, dataset, percentage, partition, "tra-l")
    tra_filename_unlabeled = 'ssl_{}/{}-ssl{}/{}-ssl{}-10-{}{}.csv'.format(percentage, dataset, percentage, dataset, percentage, partition, "tra-u")
    
    print("Converting {}".format(tra_filename))
    df_t = pd.read_csv(tra_filename)
    
    df_l = df_t[df_t[df_t.columns[-1]] != "unlabeled"]
    df_u = df_t[df_t[df_t.columns[-1]] == "unlabeled"]
    
    df_a = pd.read_csv(trs_filename)
    
    df_m = pd.merge(df_u, df_a, on=list(df_u.columns[0:-1]), how='left')
    df_m.rename(columns={"{}_x".format(df_u.columns[-1]): df_u.columns[-1], "{}_y".format(df_u.columns[-1]): "{}_answer".format(df_u.columns[-1])}, inplace=True)
    
    df_l.to_csv(tra_filename_labeled, index = False)
    df_m.to_csv(tra_filename_unlabeled, index = False)
    print("Saved files {}  {}".format(tra_filename_labeled, tra_filename_unlabeled))

def one_hot_encode(d):
    dfs = []
    lens = []
    for p in PERCENTAGES:
        for par in PARTITIONS:
            for t in TYPES:
                filename = 'ssl_{}/{}-ssl{}/{}-ssl{}-10-{}{}.csv'.format(p, d, p, d, p, par, t)
                df = pd.read_csv(filename)
                if t == "tra-u":
                    df = df.iloc[:,0:df.shape[1]-2]
                else:
                    df = df.iloc[:,0:df.shape[1]-1]
                dfs.append(df)
                lens.append(df.shape[0])
    print(len(lens))
    df = dfs[0]
    for i in range(1, len(dfs)):
        df = df.append(dfs[i], ignore_index=True)
    columns = []
    for c in df.columns:
        if df[c].dtype != 'float64' and df[c].dtype != 'int64':
            columns.append(c)
    df = pd.get_dummies(df,prefix=columns,columns=columns)

    start = 0
    end = lens[0] - 1
    i = 0
    lens.append(0)
    for p in PERCENTAGES:
        for par in PARTITIONS:
            for t in ["tra-l", "tst", "tra-u"]:
                filename = 'ssl_{}/{}-ssl{}/{}-ssl{}-10-{}{}.csv'.format(p, d, p, d, p, par, t)
                new_filename = 'ssl_{}/{}-ssl{}/{}-ssl{}-10-{}{}.csv'.format(p, d, p, d, p, par, t)
                df_o = pd.read_csv(filename)
                df_n = df.iloc[start:end+1,:]
                columns = df_o.columns.tolist()
                if t == "tra-u":
                    df_n[columns[-2]] = list(df_o[columns[-2]])
                    df_n[columns[-1]] = list(df_o[columns[-1]])
                else:
                    df_n[columns[-1]] = list(df_o[columns[-1]])
                df_n.to_csv(new_filename, index=False)
                print("Saved file {}".format(new_filename))
                print(i)
                i += 1
                start = end + 1
                end = start + lens[i] - 1

def normalize(d):
    dfs = []
    lens = []
    for p in PERCENTAGES:
        for par in PARTITIONS:
            for t in ["tra-l", "tst", "tra-u"]:
                filename = 'ssl_{}/{}-ssl{}/{}-ssl{}-10-{}{}.csv'.format(p, d, p, d, p, par, t)
                df = pd.read_csv(filename)
                if t == "tra-u":
                    df = df.iloc[:,0:df.shape[1]-2]
                else:
                    df = df.iloc[:,0:df.shape[1]-1]
                dfs.append(df)
                lens.append(df.shape[0])
    print(len(lens))
    df = dfs[0]
    for i in range(1, len(dfs)):
        df = df.append(dfs[i], ignore_index=True)
    scaler = MinMaxScaler()
    df[df.columns.tolist()] = scaler.fit_transform(df[df.columns.tolist()])

    start = 0
    end = lens[0] - 1
    i = 0
    lens.append(0)
    for p in PERCENTAGES:
        for par in PARTITIONS:
            for t in TYPES:
                filename = 'ssl_{}/{}-ssl{}/{}-ssl{}-10-{}{}.csv'.format(p, d, p, d, p, par, t)
                new_filename = 'ssl_{}/{}-ssl{}/{}-ssl{}-10-{}{}.csv'.format(p, d, p, d, p, par, t)
                df_o = pd.read_csv(filename)
                df_n = df.iloc[start:end+1,:]
                columns = df_o.columns.tolist()
                if t == "tra-u":
                    df_n[columns[-2]] = list(df_o[columns[-2]])
                    df_n[columns[-1]] = list(df_o[columns[-1]])
                else:
                    df_n[columns[-1]] = list(df_o[columns[-1]])
                df_n.to_csv(new_filename, index=False)
                print("Saved file {}".format(new_filename))
                print(i)
                i += 1
                start = end + 1
                end = start + lens[i] - 1

if __name__ == "__main__":
    for d in DATASETS:
        for p in PERCENTAGES:
            for par in PARTITIONS:
                for t in TYPES_OF_DATA:
                    convert_to_csv(d, p, par, t)

    for d in DATASETS:
        for p in PERCENTAGES:
            for par in PARTITIONS:
                separate_labeled_unlabeled(d, p, par)

    for d in DATASETS_CATEGORICAL:
        one_hot_encode(d)

    for d in DATASETS:
        normalize(d)

