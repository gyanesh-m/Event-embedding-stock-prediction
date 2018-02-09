import pandas as pd
import spacy
import numpy as np
import pickle
import spacy
import numpy as np
names=["large","mid","small"]
vectors={"large":None,"mid":None,"small":None}
final_data={"large":None,"mid":None,"small":None}
pricename={'large':'','mid':'mid','small':'small'}
for name in vectors:
    with open(name+"-vec.pkl","rb")as f:
        vectors[name]=pickle.load(f)


        
for name in names:
    
    final_data[name]={"long":[],"medium":[],"short":[]}
    df=pd.read_csv('./'+name+'-final.csv')
    df2=pd.read_csv("./nifty"+pricename[name]+"--50.csv")
    price=df2.sort_values(["date"])

    sortdf=df.sort_values(['date'])
    # makes all news of same date mapped to its corresponding date
    data={}
    corrpt={}
    
    #creates y train/test
    change_price=[]
    for i in range(len(price)-1):
        change_price.append(1*(float(price.iloc[i+1]['Change-close'])>0))
        
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
    st=[]
    mt=[]
    lt=[]
    mapping={"long":lt,"medium":mt,"short":st}
    uniq_dt=price['date'].unique()
    for i,dte in enumerate(uniq_dt):
        if(i<len(uniq_dt)-1):
            temp=[]
            st.append(dte)
            if(i-7>0):
                mt.append(uniq_dt[i-7:i])
            else:
                mt.append(uniq_dt[:i])
            if(i-30>0):
                lt.append(uniq_dt[i-30:i])
            else:
                lt.append(uniq_dt[:i])
    
   
    print("Doing for "+name)
    for term in mapping:
        count=0
        for dat in mapping[term]:
            #print(count)
            temp=[]
            if(type(dat)!=str):
                for entry in dat:
                    try:
                        temp.append(vectors[name][entry])
                    except Exception as e:
                        pass
                if(len(temp)!=0):
                    final_data[name][term].append(np.array(temp))
                else:
                    final_data[name][term].append(np.zeros(300))
            else:
                try:
                    final_data[name][term].append(vectors[name][dat])
                except Exception as e:
                    print("not for "+dat)
                    final_data[name][term].append(np.zeros(300))
            count+=1
    print("done "+name)
    print("Summary for "+name)
    req=0
    for term in final_data[name]:
        print(len(final_data[name][term]))
        for line,linedata in enumerate(final_data[name][term]):
            if term=='long':
               print(len(linedata))
               print(linedata.shape)
               if(linedata.shape[0]==300):
                   req=29
               else:
                   req=30-len(linedata)
            elif term=='medium':
               print(len(linedata))
               req=7-len(linedata)
            if term!='short':
                print(req)
                final_data[name][term][line]=np.hstack((linedata,np.zeros((req,300))))
    for term in final_data[name]:
        final_data[name][term]=np.array(final_data[name][term])
        print("shape for "+term+str(final_data[name][term].shape))
    print("Length of the price file",len(change_price))
    dt={"X":final_data[name],"Y":change_price}
    
    with open(name+"-lms_vec_training.pkl","wb")as f:
        pickle.dump(dt,f)
