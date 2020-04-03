import pandas as pd

DATASETS = ['satimage',
 'glass',
 'hepatitis',
 'magic',
 'bupa',
 'wisconsin',
 'lymphography',
 'phoneme',
 'titanic',
 'abalone',
 'spambase',
 'sonar',
 'wine',
 'vowel',
 'australian',
 'vehicle',
 'spectfheart',
 'contraceptive',
 'twonorm',
 'breast',
 'crx',
 'ecoli',
 'saheart',
 'automobile',
 'segment',
 'penbased',
 'flare',
 'movement_libras',
 'banana',
 'coil2000',
 'tic-tac-toe',
 'ring',
 'dermatology',
 'page-blocks',
 'thyroid',
 'mammographic',
 'iris',
 'german',
 'cleveland',
 'haberman',
 'monk-2',
 'nursery',
 'appendicitis',
 'zoo',
 'pima',
 'chess',
 'splice',
 'mushroom',
 'heart',
 'tae',
 'led7digit',
 'yeast',
 'marketing',
 'housevotes',
 'texture']

PERCENTAGES = [10, 20, 30, 40]

PARTITIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

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
    
if __name__ == "__main__":
    for d in DATASETS:
        for p in PERCENTAGES:
            for par in PARTITIONS:
                separate_labeled_unlabeled(d, p, par)