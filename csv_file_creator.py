import os
import pandas as pd
import numpy as np
import random as rnd
import shutil
import sys

if len(sys.argv)!=2:
    print("Make sure following directories exist before running this file\n1.'Input'\n2.'Output'")
    print("\nUsage: python csv_file_creator.py <train_test_split_fraction>\nE.g.: python csv_file_creator.py 0.8")
    sys.exit("Exiting...")

if float(sys.argv[1])>1 or float(sys.argv[1])<0:
    print("Train Test Split fraction out of bounds!")
    sys.exit("Exiting...")
    
f=np.array([])
for root, dirs, files in os.walk(os.path.join(os.getcwd(),"Input")):
    f = np.append(f,np.ravel(np.array(files)))


f_drowsy=np.array([])
f_alert=np.array([])
for names in f:
    name=names.split(".")[0]
    if names.find("Drowsy")!=-1:
        f_drowsy=np.append(f_drowsy,name)
    elif names.find("Alert")!=-1:
        f_alert=np.append(f_alert,name)
    
rnd.shuffle(f_drowsy)
rnd.shuffle(f_alert)

total=len(f_drowsy)+len(f_alert)

train_test_split=float(sys.argv[1])
train_test=np.array([])
for i in range(int(train_test_split*total)):
    train_test=np.append(train_test,'training')

for i in range(total-int(train_test_split*total)):
    train_test=np.append(train_test,'testing')

names1=np.append(f_drowsy[:int(train_test_split*len(f_drowsy))],f_alert[:int(train_test_split*len(f_alert))])
names2=np.append(f_drowsy[int(train_test_split*len(f_drowsy)):],f_alert[int(train_test_split*len(f_alert)):])
names=np.append(names1,names2)

types1=np.append(['Drowsy']*int(train_test_split*len(f_drowsy)),['Alert']*int(train_test_split*len(f_alert)))
types2=np.append(['Drowsy']*(len(f_drowsy)-int(train_test_split*len(f_drowsy))),['Alert']*(len(f_alert)-int(train_test_split*len(f_alert))))
types=np.append(types1,types2)


ret = pd.DataFrame(train_test.reshape(-1))
ret['type']= types.reshape(-1)
ret['names']=names.reshape(-1)

print("###############\n")
print("Creating 'data' directory (Will delete if already exists!)...")
dir = os.path.join(os.getcwd(),"data")
if os.path.exists(dir):
    shutil.rmtree(dir)
os.mkdir(dir)
print("Created 'data' directory successfully!")

ret.to_csv(os.path.join(dir,'data_file.csv'),index=False,header=False)

print("\nSaved data_file.csv to data directory")
print("\n###############")

print("\nCreating 'training' and 'testing' directory inside 'data' directory...\n")
traindir=os.path.join(dir,"training")
os.mkdir(traindir)
testdir=os.path.join(dir,"testing")
os.mkdir(testdir)
seqdir=os.path.join(dir,"sequences")
os.mkdir(seqdir)
seqtrain=os.path.join(dir,"sequences", "training")
os.mkdir(seqtrain)
seqtest=os.path.join(dir,"sequences", "testing")
os.mkdir(seqtest)
print("Created directories successfully!")


print("\nCreating 'Alert' and 'Drowsy' directory inside 'training' and 'testing' directories...\n")
os.mkdir(os.path.join(traindir,"Alert"))
os.mkdir(os.path.join(testdir,"Alert"))
os.mkdir(os.path.join(traindir,"Drowsy"))
os.mkdir(os.path.join(testdir,"Drowsy"))
print("Created directories successfully!")

import pandas as pd
import os
import shutil
from functools import reduce


df = pd.read_csv("./data/data_file.csv", header = None)


df_train = df[df.iloc[:,0] == 'training']
df_test = df[df.iloc[:,0] == 'testing']

def move(df,type1):
    for i in range(df.shape[0]):
        cat = df.iloc[i,2].strip().split('_')[0]
        num = df.iloc[i,2].strip().split('_')[1][-1]
        strng = df.iloc[i,2]
        shutil.move(reduce(os.path.join,[os.getcwd(),"Output",num,cat]),reduce(os.path.join,[os.getcwd(),"data",type1,cat,strng]))
        source = reduce(os.path.join,[os.getcwd(),"data",type1,cat,strng])
        destination = reduce(os.path.join,[os.getcwd(),"data",type1,cat])
        for files in os.listdir(source):
            if files.endswith(".jpg"):
                shutil.copy(os.path.join(source,files),destination)


move(df_train,"training")
move(df_test,"testing")

#def move_temp(df,type1):
#    for i in range(df.shape[0]):
#s        cat = df.iloc[i,2].strip().split('_')[0]
#        num = df.iloc[i,2].strip().split('_')[1][-1]
#        strng = df.iloc[i,2]
#        shutil.copy(reduce(os.path.join,[os.getcwd(),"data",type1,strng]),reduce(os.path.join,[os.getcwd(),"data",type1]))


