from django.shortcuts import render

def Index(request):
    return render(request,'Index.html')

import pandas as pd
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
# Create your views here.

def fakebill(request):
    if(request.method=='POST'):
        data=request.POST
        dg=data.get('diagonal')
        hl=data.get('height_left')
        hr=data.get('height_right')
        ml=data.get('margin_low')
        mu=data.get('margin_up')
        lg=data.get('length')
        path="C:/Users/KAVYA/Desktop/InternshipProject/26_fakebillprediction/fake_bills.csv"
        data=pd.read_csv(path)
        le=LabelEncoder()
        data['is_genuine']=le.fit_transform(data['is_genuine'])
        df=data['margin_low'].mean()
        data['margin_low'].fillna(value=df,axis=0,inplace=True)
        data.isna().sum()
        inputs=data.drop('is_genuine','columns')
        output=data['is_genuine']
        x_train, x_test, y_train, y_test = train_test_split(inputs, output, test_size=0.2)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        model=LogisticRegression()
        model.fit(x_train,y_train)
        ip=scaler.transform([[dg,hl,hr,ml,mu,lg]])
        result = model.predict(ip)
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        acc=result[0],acc*100
        if (result==1):
            result="Bill is genuine"
        else:
            result="Bill is fake(Not genuine)"

        return render(request,'fakebill.html',context={'result':result,'acc':acc})
    
    return render(request,'fakebill.html')
   

def employee(request):
    if(request.method=="POST"):
        data=request.POST
        stag=data.get('stag')
        age=data.get('age')
        id=data.get('industry')
        prof=data.get('profession')
        tr=data.get('traffic')
        ch=data.get('coach')
        gw=data.get('greywage')
        way=data.get('way')
        ext=data.get('extraversion')
        idp=data.get('independ')
        sc=data.get('selfcontrol')
        ax=data.get('anxiety')
        nv=data.get('novator')
        path="C:/Users/KAVYA/Desktop/InternshipProject/27_EmployeeQuittingtheirJobPrediction/turnover.csv"
        data=pd.read_csv(path)
        le=LabelEncoder()
        data['gender']=le.fit_transform(data['gender'])
        data['industry']=le.fit_transform(data['industry'])
        data['profession']=le.fit_transform(data['profession'])
        data['traffic']=le.fit_transform(data['traffic'])
        data['coach']=le.fit_transform(data['coach'])
        data['head_gender']=le.fit_transform(data['head_gender'])
        data['greywage']=le.fit_transform(data['greywage'])
        data['way']=le.fit_transform(data['way'])
        inputs = data.drop(['event','gender','head_gender'],'columns')
        output = data['event'] 
        x_train,x_test,y_train,y_test=train_test_split(inputs,output,train_size=0.8)
        knn = KNeighborsClassifier(n_neighbors=33)
        # Fit the classifier to the training data 
        knn.fit(x_train,y_train)
        # Make predictions on the test data 
        y_pred = knn.predict(x_test)
        result=knn.predict([[stag,age,id,prof,tr,ch,gw,way,ext,idp,sc,ax,nv]])
        acc = accuracy_score(y_test, y_pred)
        acc=result[0],acc*100
        if (result==0):
            result="Employee will quit the job"
        else:
            result="Employee will not quit the job"
        return render(request,"employee.html",context={'result':result,'acc':acc})
        
    return render(request,'employee.html')


# Create your views here.
