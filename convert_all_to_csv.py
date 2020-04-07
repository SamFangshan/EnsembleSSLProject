DATASETS = ['satimage','glass','hepatitis', 'magic', 'bupa', 'wisconsin', 'lymphography', 'phoneme', 'titanic', 'abalone', 'spambase', 'sonar', 'wine', 'vowel', 'australian', 'vehicle', 'spectfheart', 'contraceptive', 'twonorm', 'breast', 'crx', 'ecoli', 'saheart', 'automobile', 'segment', 'penbased', 'flare', 'movement_libras', 'banana', 'coil2000', 'tic-tac-toe', 'ring', 'dermatology', 'page-blocks', 'thyroid', 'mammographic', 'iris', 'german', 'cleveland', 'haberman', 'monk-2', 'nursery', 'appendicitis', 'zoo', 'pima', 'chess', 'splice', 'mushroom', 'heart', 'tae', 'led7digit', 'yeast', 'marketing', 'housevotes', 'texture']

PERCENTAGES = [10, 20, 30, 40]

PARTITIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

TYPES_OF_DATA = ['tra', 'trs', 'tst']

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

if __name__ == "__main__":
    for d in DATASETS:
        for p in PERCENTAGES:
            for par in PARTITIONS:
                for t in TYPES_OF_DATA:
                    convert_to_csv(d, p, par, t)
