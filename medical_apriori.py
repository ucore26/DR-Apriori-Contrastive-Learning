import pandas as pd
import itertools

min_support = 0.3
min_confidence = 0.8

def loadDataSet(filename):
    datas = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            datas.append(line.strip().split(','))
    return datas

def count(items, datas):
    num = 0
    for data in datas:
        if set(items).issubset(set(data)):
            num += 1
    return num

def getFleft(items, datas):
    fl = []
    sl = []
    for item in items:
        support = count(item, datas) / float(len(datas))
        if support > 0:
            if sorted(item) not in fl:
                fl.append(sorted(item))
                sl.append(support)
    return fl, sl

def generate_Fleft(items, datas):
    f_s = {}
    f_left = []
    fl, sl = getFleft(items, datas)
    for f, s in zip(fl, sl):
        f_left.append(f)
        f_s[frozenset(f)] = s
    return f_left, f_s

class cartesian(object):
    def __init__(self):
        self._data_list = []

    def add_data(self, data=[]):
        self._data_list.append(data)

    def build(self):
        res = []
        for item in itertools.product(*self._data_list):
            res.append(item)
        return res

def generate_rule(fm, sm, left_s, f_er, f_s_class):
    rules = []
    for f, s in zip(fm, sm):
        rules.extend(rule(f, s, left_s, f_er, f_s_class, []))
    return rules

def rule(f, s, left_s, f_er, f_s_class, cur_rule):
    lift = f_s_class[frozenset(f[0])] / left_s[frozenset(f[0])]
    strength = (2 * f_er[frozenset(f[0])] * lift) / (f_er[frozenset(f[0])] + lift)
    if lift >= 1:
        cur_rule.append((f[0], f[1], f_er[frozenset(f[0])], lift, strength))
    return cur_rule

def main():
    df = pd.read_excel("./data/2020.7.11_fenji.xlsx", header=0, dtype=str)
    df1 = df[df['47术后是否感染'] == '471']
    df2 = df[df['47术后是否感染'] == '472']
    datas = df.values.tolist()
    datas1 = df1.values.tolist()
    datas2 = df2.values.tolist()

    fk_new = []
    with open("./data/fk_left_neg.txt", 'r', encoding='utf-8') as fr:
        for line in fr:
            fk_new.append(line.strip().split(' '))

    fk_positive, f_s_positive = generate_Fleft(fk_new, datas2)
    print("频繁项集：{} 个".format(len(f_s_positive)))
    for key, value in f_s_positive.items():
        print("{} : {:.2f}".format(key, value))

    fk_negative, f_s_negative = generate_Fleft(fk_positive, datas1)
    print("频繁项集：{} 个".format(len(f_s_negative)))
    for key, value in f_s_negative.items():
        print("{} : {:.2f}".format(key, value))

    fk_final = []
    f_er = {}
    min_er = 1
    for item in fk_negative:
        if f_s_positive[frozenset(item)] / f_s_negative[frozenset(item)] >= min_er:
            fk_final.append(item)
            f_er[frozenset(item)] = f_s_positive[frozenset(item)] / f_s_negative[frozenset(item)]
    print("频繁项集：{} 个".format(len(fk_final)))
    for key, value in f_er.items():
        print("{} : {:.2f}".format(key, value))

    fk_all, f_s_all = generate_Fleft(fk_final, datas)
    print("频繁项集：{} 个".format(len(f_s_all)))
    for key, value in f_s_all.items():
        print("{} : {:.2f}".format(key, value))

    fk_final = []
    f_er = {}
    min_er = 1
    for item in fk_all:
        if f_s_positive[frozenset(item)] / f_s_all[frozenset(item)] >= min_er:
            fk_final.append(item)
            f_er[frozenset(item)] = f_s_positive[frozenset(item)] / f_s_all[frozenset(item)]
    print("频繁项集：{} 个".format(len(fk_final)))
    for key, value in f_er.items():
        print("{} : {:.2f}".format(key, value))

    right = [['472']]
    car = cartesian()
    car.add_data(fk_final)
    car.add_data(right)
    issues = car.build()
    print(len(issues))

    issue_supp = []
    for items in issues:
        supp = count(items[0] + items[1], datas) / len(datas)
        issue_supp.append(supp)

    rules = generate_rule(issues, issue_supp, f_s_all, f_er, f_s_positive)
    print("关联规则：{}个".format(len(rules)))

    new_rules = []
    for item in rules:
        new_rules.append(item[::-1])
    new_rules = sorted(new_rules, reverse=True)
    print(new_rules)

    count_num = 0
    for strength, lift, er, result, reason in new_rules[:100]:
        if "301" in reason:
            print("{} ----> {} : {:.4f},{:.4f},{:.4f}".format(reason, result, er, lift, strength))
            count_num += 1
    print("感染关联规则：{}个".format(count_num))

    for item in rules:
        print(item)

if __name__ == "__main__":
    main()
