
import pandas as pd
import numpy as np

dataset=pd.read_csv('telco-customer-churn.csv')

from sklearn.cross_validation import train_test_split
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=0)

X_train=np.delete(X_train,0,axis=1)
X_train=pd.DataFrame(X_train)


X_train[0]=X_train[0].replace(['Female','Male'],['0','1'])
X_train[6]=X_train[6].replace(['No phone service'],['No'])
for i in range(8,14):
    X_train[i]=X_train[i].replace(['No internet service'],['No'])


X_train=X_train.iloc[:,:].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
one=OneHotEncoder(categorical_features=[7])
one2=OneHotEncoder(categorical_features=[15])
le=LabelEncoder()
X_train[:,0]=le.fit_transform(X_train[:,0])
X_train[:,2]=le.fit_transform(X_train[:,2])
X_train[:,3]=le.fit_transform(X_train[:,3])
X_train[:,5]=le.fit_transform(X_train[:,5])
X_train[:,6]=le.fit_transform(X_train[:,6])
X_train[:,7]=le.fit_transform(X_train[:,7])
X_train[:,8]=le.fit_transform(X_train[:,8])
X_train[:,9]=le.fit_transform(X_train[:,9])
X_train[:,10]=le.fit_transform(X_train[:,10])
X_train[:,11]=le.fit_transform(X_train[:,11])
X_train[:,12]=le.fit_transform(X_train[:,12])
X_train[:,13]=le.fit_transform(X_train[:,13])
X_train[:,14]=le.fit_transform(X_train[:,14])
X_train[:,15]=le.fit_transform(X_train[:,15])
X_train[:,16]=le.fit_transform(X_train[:,16])

X_train=pd.DataFrame(X_train)

X_train.iloc[786,18]='100'
X_train.iloc[1029,18]='100'
X_train.iloc[1757,18]='100'
X_train.iloc[2721,18]='100'
X_train.iloc[3606,18]='100'
X_train.iloc[4114,18]='100'
X_train.iloc[4896,18]='100'
X_train.iloc[5145,18]='100'
X_train=one.fit_transform(X_train).toarray()
X_train=X_train[:,1:]

X_train=one2.fit_transform(X_train).toarray()
X_train=X_train[:,1:]
X_train=np.delete(X_train,18,axis=1)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)

X_test=np.delete(X_test,0,axis=1)
X_test=pd.DataFrame(X_test)


X_test[0]=X_test[0].replace(['Female','Male'],['0','1'])
X_test[6]=X_test[6].replace(['No phone service'],['No'])
for i in range(8,14):
    X_test[i]=X_test[i].replace(['No internet service'],['No'])

X_test=X_test.iloc[:,:].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
one3=OneHotEncoder(categorical_features=[7])
one4=OneHotEncoder(categorical_features=[15])
le=LabelEncoder()
X_test[:,0]=le.fit_transform(X_test[:,0])
X_test[:,2]=le.fit_transform(X_test[:,2])
X_test[:,3]=le.fit_transform(X_test[:,3])
X_test[:,5]=le.fit_transform(X_test[:,5])
X_test[:,6]=le.fit_transform(X_test[:,6])
X_test[:,7]=le.fit_transform(X_test[:,7])
X_test[:,8]=le.fit_transform(X_test[:,8])
X_test[:,9]=le.fit_transform(X_test[:,9])
X_test[:,10]=le.fit_transform(X_test[:,10])
X_test[:,11]=le.fit_transform(X_test[:,11])
X_test[:,12]=le.fit_transform(X_test[:,12])
X_test[:,13]=le.fit_transform(X_test[:,13])
X_test[:,14]=le.fit_transform(X_test[:,14])
X_test[:,15]=le.fit_transform(X_test[:,15])
X_test[:,16]=le.fit_transform(X_test[:,16])

X_test=pd.DataFrame(X_test)

X_test.iloc[212,18]='100'
X_test.iloc[404,18]='100'
X_test.iloc[1039,18]='100'

X_test=one3.fit_transform(X_test).toarray()
X_test=X_test[:,1:]

X_test=one4.fit_transform(X_test).toarray()
X_test=X_test[:,1:]

X_test=np.delete(X_test,18,axis=1)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_test=sc.fit_transform(X_test)



y_train[:,0]=le.fit_transform(y_train[:,0])
y_test[:,0]=le.fit_transform(y_test[:,0])
#y_pred[:,0]=le.fit_transform(y_pred[:,0])

from keras.models import Sequential
from keras.layers import Dense

seq=Sequential()
seq.add(Dense(output_dim=11,init='uniform',activation='relu',input_dim=20))
seq.add(Dense(output_dim=11,init='uniform',activation='relu'))
seq.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
seq.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

seq.fit(X_train,y_train,batch_size=10,epochs=100 )

y_pred=seq.predict(X_test)
y_test=(y_test>0.5)
y_pred=(y_pred>0.5)
y_pred=y_pred*1
y_test=y_test*1


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


X_train=X_train.iloc[:,:].values
y_pred=y_pred.iloc[:,:].values
y_test=y_test.iloc[:,:].values
y_train=y_train.iloc[:,:].values

X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)
y_train=pd.DataFrame(y_train)
y_test=pd.DataFrame(y_test)

y_pred=pd.DataFrame(data=y_pred)
