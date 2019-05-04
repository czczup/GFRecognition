import time
import numpy as np
import sys
import datetime
import pandas as pd
import os



date2position = {}
datestr2dateint = {}
str2int = {}
for i in range(24):
    str2int[str(i).zfill(2)] = i

for i in range(182):
    date = datetime.date(day=1, month=10, year=2018)+datetime.timedelta(days=i)
    date_int = int(date.__str__().replace("-", ""))
    date2position[date_int] = i
    datestr2dateint[str(date_int)] = date_int


def calculate_visit_list(visit_lst):
    ys, zs = [], []  # 开始时间，持续时间
    count = 0
    start = visit_lst[0]
    for i in range(len(visit_lst)-1):
        if str2int[visit_lst[i]] == str2int[visit_lst[i+1]]-1:
            count += 1
        else:
            ys.append(str2int[start])
            zs.append(count)
            start = visit_lst[i+1]
            count = 0
    else:
        ys.append(str2int[start])
        zs.append(count)
    return ys, zs


def visit2array(table):
    strings = table[1]
    init = np.zeros((182, 24, 24), dtype=np.uint8)
    for string in strings:
        temp = []
        for item in string.split(','):
            temp.append([item[0:8], item[9:].split("|")])
        for date, visit_lst in temp:
            x = date2position[datestr2dateint[date]]
            ys, zs = calculate_visit_list(visit_lst)
            for i in range(len(ys)):
                y = ys[i]
                z = zs[i]
                init[x][y][z] += 1
    return init


def visit2array_test():
    start_time = time.time()
    for i in range(0, 10000):
        filename = str(i).zfill(6)
        table = pd.read_table("../data/test_visit/test/"+filename+".txt", header=None)
        array = visit2array(table)
        np.save("../data/npy/test_visit2/"+filename+".npy", array)
        sys.stdout.write('\r>> Processing visit data %d/%d'%(i+1, 10000))
        sys.stdout.flush()
    sys.stdout.write('\n')
    print("using time:%.2fs"%(time.time()-start_time))


def visit2array_train():
    table = pd.read_csv("../data/train.txt", header=None)
    filenames = [a[0].split("/")[-1].split('.')[0] for a in table.values]
    length = len(filenames)
    start_time = time.time()
    for index, filename in enumerate(filenames):
        table = pd.read_table("../data/train_visit/"+filename+".txt", header=None)
        array = visit2array(table)
        np.save("../data/npy/train_visit2/"+filename+".npy", array)
        sys.stdout.write('\r>> Processing visit data %d/%d'%(index+1, length))
        sys.stdout.flush()
    sys.stdout.write('\n')
    print("using time:%.2fs"%(time.time()-start_time))


def visit2array_valid():
    table = pd.read_csv("../data/valid.txt", header=None)
    filenames = [a[0].split("/")[-1].split('.')[0] for a in table.values]
    length = len(filenames)
    start_time = time.time()
    for index, filename in enumerate(filenames):
        table = pd.read_table("../data/train_visit/"+filename+".txt", header=None)
        array = visit2array(table)
        np.save("../data/npy/train_visit2/"+filename+".npy", array)
        sys.stdout.write('\r>> Processing visit data %d/%d'%(index+1, length))
        sys.stdout.flush()
    sys.stdout.write('\n')
    print("using time:%.2fs"%(time.time()-start_time))


if __name__ == '__main__':

    if not os.path.exists("../data/npy/test_visit2/"):
        os.makedirs("../data/npy/test_visit2/")
    if not os.path.exists("../data/npy/train_visit2/"):
        os.makedirs("../data/npy/train_visit2/")
    visit2array_train()
    visit2array_valid()
    visit2array_test()

