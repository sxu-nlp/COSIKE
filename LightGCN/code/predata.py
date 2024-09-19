import pandas as pd
import numpy as np
import random

filename = "/home/stu4/project/lightgcn/data/meld_word/knowledge/relation.txt"

data = pd.read_csv(filename,header=None).to_numpy()

train_file = "/home/stu4/project/lightgcn/data/meld_word/knowledge/train.txt"
test_file = "/home/stu4/project/lightgcn/data/meld_word/knowledge/test.txt"

length = len(data)
index = [i for i in range(length)]
np.random.shuffle(index)

train_index = index[:int(length*0.8)]
test_index = index[int(length*0.8):]

train_data = data[train_index]
test_data = data[test_index]


with open(train_file, 'w') as file:
    for row in train_data:
        user, att_user, re = row[0].strip().split('\t')
        print(row[0])
        file.write(f"{user} {att_user} {re}\n")
    print('write train file over!')

# 写入测试集文件
with open(test_file, 'w') as file:
    for row in test_data:
        user, att_user, re = row[0].strip().split('\t')
        file.write(f"{user} {att_user} {re}\n")
    print('write test file over!')




'''with open(train_file,'w') as file:
    for user,att_user in train_data:
        file.write(str(user)+' '+str(att_user)+'\n')
    print('write over!')

with open(test_file,'w') as file:
    for user,att_user in test_data:
        file.write(str(user)+' '+str(att_user)+'\n')
    print("write over!")'''
