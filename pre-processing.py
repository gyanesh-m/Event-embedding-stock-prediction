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
    print("Setting up the change in price for "+name)
    for i in range(len(price)-1):
        change_price.append(1*(float(price.iloc[i+1]['Change-close'])>0))
    print("Merging data for same date for "+name)
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
        #The data is already added before , so nothing to worry about.
        except Exception as e:
            print(e)
        data[sortdf.iloc[i]['date']].append(' '.join(temp))
    
    for i in data:
        data[i]=set(data[i])
    st=[]
    mt=[]
    lt=[]
    mapping={"long":lt,"medium":mt,"short":st}
    uniq_dt=price['date'].unique()
    for i in range(len(uniq_dt)):
        #last but one since the next day is always for prediction.
        if(i<len(uniq_dt)-1):
            st.append(uniq_dt[i])
            if(i-7>=0):
                mt.append(uniq_dt[i-7:i])
            else:
                mt.append(uniq_dt[:i+1])
            if(i-30>=0):
                lt.append(uniq_dt[i-30:i])
            else:
                lt.append(uniq_dt[:i+1])
    print("Doing for "+name)
    
    #here term is for the time period, long, medium and short!
    for term in mapping:
        count=0
        for dat in mapping[term]:
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
                    final_data[name][term].append(np.zeros(300))
            count+=1

    for term in ['long','medium','short']:
        if(term=='long'):
            wall=30
        elif(term=='medium'):
            wall=7
        else:
            wall=1
        print("Resizing all vectors to same size for "  +name+" "+term)
        for dat in range(len(final_data[name][term])):
            ref=len(final_data[name][term][dat])
            if(ref==300):
                final_data[name][term][dat]=np.zeros((wall,300))
                continue
            else:
                if(wall-ref>0):
                    final_data[name][term][dat]=np.vstack((final_data[name][term][dat],np.zeros((wall-ref,300))))
    print("Shapes after resize")
    for term in final_data[name]:
        final_data[name][term]=np.array(final_data[name][term])
        print(" for "+term+"-"+str(final_data[name][term].shape))
    print("Length of the change in price for "+name,len(change_price))
    dt={"X":final_data[name],"Y":np.array(change_price)}
    with open(name+"-lms_vec_training.pkl","wb")as f:
        pickle.dump(dt,f)
    print("done for "+name)
    print("#-"*15)
print("Pre processing done !")