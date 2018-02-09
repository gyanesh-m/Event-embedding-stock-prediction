import pandas as pd
import spacy
import numpy as np
import pickle
import spacy
import numpy as np
names=["large","mid","small"]
for name in names:
    df=pd.read_csv('./'+name'-final.csv')
    sortdf=df.sort_values(['date'])
    # makes all news of same date mapped to its corresponding date
    data={}
    corrpt={}
    for i in range(len(sortdf)):
        temp=[]
        if(data.get(sortdf.iloc[i]['date'],None)==None):
            data[sortdf.iloc[i]['date']]=[]
        temp.append(str(sortdf.iloc[i]['data']).strip("\n"))
        try:
            if('http' not in sortdf.iloc[i]['url'] ):
                try:
                    corrpt[sortdf.iloc[i]['date']].append(sortdf.iloc[i]['url'])
                    temp.append(str(sortdf.iloc[i]['url']).strip("\n"))
                except:
                    corrpt[sortdf.iloc[i]['date']]=[]
                    corrpt[sortdf.iloc[i]['date']].append(sortdf.iloc[i]['url'].rstrip("\n"))
                    temp.append(str(sortdf.iloc[i]['url']).strip("\n"))
        except Exception as e:
            print(e)
            print("No worries, Taken care before")
        data[sortdf.iloc[i]['date']].append(' '.join(temp))
    for i in data:
        data[i]=set(data[i])
    sortdf.iloc[1]['date']
    len(sortdf)
    # data


    model=spacy.load("en_vectors_web_lg")
    data_vec={}
    count=0
    for date in data:
        temp=[]
        #generates mean values for all the vectors
        for text in data[date]:
            doc=model(text)
            temp.append(doc.vector)
        #map the vector mean to that day
        meank=np.mean(np.array(temp,dtype=np.float32),axis=0)
        data_vec[date]=meank
        count+=1
        print(count)
    with open(name+"-vec.pkl","wb") as f:
        pickle.dump(data_vec,f)

