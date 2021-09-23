import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
%matplotlib inline
data=pd.read_csv('mnist.csv')
data.head()
#extracting data from dataset and viewing them up close
a= data.iloc[1, 2:].values
#reshapinig data to a reasonable size
a= a.reshape(28,28).astype('uint8')
plt.imshow(a)
#preparing data 
#separating labels and data values
df_x=data.iloc[:,2:]
df_y=data.iloc[:,1]
#creating test and train sizes/batches
x_train,x_test,y_train,y_test =train_test_split(df_x, df_y, test_size=0.2 ,random_state=4)
#check data
y_train.head()
#rf classifier
rf=RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
pred=rf.predict(x_test)
s=y_test.values
#calculating number of values that are equal in both the cases
count=0
for i in range(len(pred)):
    if(s[i]==pred[i]):
        count+=1
#checking the accuracy
count/(len(pred))
