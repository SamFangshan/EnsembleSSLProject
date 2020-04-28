"""
Execute this code to remove predict_proba() condition of scikit-learn voting classifier,
to be able to call predict_proba() even when 'hard' vote is used.
"""
def cheat():
    f = open("/usr/local/lib/python3.6/site-packages/sklearn/ensemble/_voting.py", "r")
    lines = f.readlines()
    f.close()
    f = open("/usr/local/lib/python3.6/site-packages/sklearn/ensemble/_voting.py", "w+")
    for i in range(len(lines)):
        if i <= 275 or i >= 279:
            f.write(lines[i])
    f.close()


if __name__ == "__main__":
    cheat()
