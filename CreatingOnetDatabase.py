__author__ = 'aalibasic'

import pandas as pd
import pickle

def run():

    #### ONET Database

    tmpDats=[]
    infiles=["db_20_0/Occupation Data.txt",]
    for infile in infiles:
        foo=pd.read_csv(infile,'\t')
        tmpDats.append(foo)

    allDat_ONET=pd.concat(tmpDats,axis=1)
    allDat_ONET=pd.DataFrame(allDat_ONET,columns=['O*NET-SOC Code','Title', 'Description'])
    allDat_ONET=allDat_ONET.dropna(subset=['Title', 'Description']).reset_index(drop=True)
    allDat_ONET['Description'] = allDat_ONET['Description'].astype(str)
    allDat_ONET['Description'] = allDat_ONET['Description'].map(str) + allDat_ONET['Title'].astype(str)

    # Tasks
    tmpDats=[]
    infiles=["db_20_0/Task Statements.txt",]
    for infile in infiles:
        foo=pd.read_csv(infile,'\t')
        tmpDats.append(foo)

    Task_ONET=pd.concat(tmpDats,axis=1)
    Task_ONET=pd.DataFrame(Task_ONET,columns=['O*NET-SOC Code','Task'])
    Task_ONET=Task_ONET.dropna(subset=['O*NET-SOC Code','Task']).reset_index(drop=True)


    # Matching tasks
    for i in range(1,len(Task_ONET)):
        temp=Task_ONET['Task'][i]
        if Task_ONET['O*NET-SOC Code'][i]== Task_ONET['O*NET-SOC Code'][i-1]:
            temp= Task_ONET['Task'][i-1] + ' ' + temp
        else:
            temp=Task_ONET['Task'][i]
        Task_ONET['Task'][i]=temp

    y=Task_ONET.drop_duplicates(subset='O*NET-SOC Code', keep='last').reset_index(drop=True)
    allDat_ONET=pd.merge(allDat_ONET, y, left_on='O*NET-SOC Code', right_on='O*NET-SOC Code', how='outer')


    allDat_ONET['Task'].fillna('',inplace=True)
    allDat_ONET['Description'] = allDat_ONET['Description'].map(str)  + ' ' + allDat_ONET['Task'].astype(str)
    del allDat_ONET['Task']

    with open('allDat_ONET.pkl', 'wb') as handle:
        pickle.dump(allDat_ONET, handle)

    writer = pd.ExcelWriter('allDat_ONET.xlsx')
    allDat_ONET.to_excel(writer,'Sheet1')
    #df2.to_excel(writer,'Sheet2')
    writer.save()
