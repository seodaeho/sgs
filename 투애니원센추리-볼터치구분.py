
# coding: utf-8

# In[1]:


from tensorflow.keras.models import model_from_json
import pickle
import time
import pymysql
import datetime
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import ast


# In[2]:


def flatten(l): 
    flatList = [] 
    for elem in l: 
        if type(elem) == list: 
            for e in elem: 
                flatList.append(e) 
        else: 
            flatList.append(elem) 
    return flatList



# In[3]:


# 분석부분


# In[4]:


# 여기부터 실행


# In[5]:


from sklearn.externals import joblib 
file_name = 'E:\\분석\\투애니원\\X.pkl' 
X = joblib.load(file_name)
file_name = 'E:\\분석\\투애니원\\Y.pkl' 
Y = joblib.load(file_name)


# In[6]:


X=X.reset_index(drop=True)
Y=Y.reset_index(drop=True)


# In[7]:


len(X)


# In[8]:


X.tail()


# In[9]:


X.tail()


# In[10]:


Y.tail()


# In[11]:


X_Y=pd.concat([X,Y], axis=1)


# In[12]:


X_Y.head(10)


# In[13]:


X_Y=X_Y.dropna().reset_index(drop=True)


# In[14]:


X_Y.head()


# In[15]:


볼터치=pd.read_excel('E:\\분석\\투애니원\\볼터치.xlsx')


# In[16]:


볼터치_o=list(볼터치.loc[볼터치['볼터치구분']==1]['유형'])


# In[17]:


볼터치_x=list(볼터치.loc[볼터치['볼터치구분']==0]['유형'])


# In[18]:


X_Y['class']=[1 if x in 볼터치_o else 0 for x in list(X_Y['class'])]


# In[40]:


X_Y['class'].value_counts()


# In[41]:


X_Y=X_Y.groupby('class').apply(lambda x: x.sample(100000))


# In[42]:


X_Y['class'].value_counts()


# In[44]:


X_Y=X_Y.reset_index(drop=True)
X_Y=X_Y.sample(frac=1).reset_index(drop=True)


# In[45]:


X_Y.head()


# In[46]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore', sparse = False)


# In[47]:


ohe.fit(X_Y[['class']])
cat_ohe = ohe.transform(X_Y[['class']])


# In[81]:


X_Y[['class']]


# In[79]:


cat_ohe


# In[48]:


X_Y.tail(100)


# In[49]:


X_Y.shape


# In[50]:


X=X_Y.iloc[:,:-1]


# In[51]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
#from tensorflow.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
model = Sequential()
model.add(Dense(1000, input_dim=(X.shape[1]), activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(len(np.unique(X_Y[['class']])), activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


# In[52]:


from sklearn.preprocessing import StandardScaler


# In[53]:


scaler=StandardScaler()
scaler.fit(X)
dataset_x = scaler.transform(X)


# In[54]:


import pickle
with open('E:\\분석\\투애니원\\scaler_볼터치', 'wb') as f:
    pickle.dump(scaler, f)
with open('E:\\분석\\투애니원\\ohe_볼터치', 'wb') as f:
    pickle.dump(ohe, f)


# In[55]:


msk = np.random.rand(len(dataset_x)) < 0.8
train_x = dataset_x[msk]
train_y= cat_ohe[msk]
test_x = dataset_x[~msk]
test_y = cat_ohe[~msk]


# In[60]:


early_stopping = EarlyStopping(monitor='val_loss', patience=150)


# In[61]:


model.fit(train_x, train_y, epochs=1000, batch_size=1500, validation_data=(test_x, test_y),verbose=2, callbacks=[early_stopping])


# In[62]:


model_json = model.to_json()
with open("E:\\분석\\투애니원\\model_볼터치.json", "w") as json_file : 
    json_file.write(model_json)
model.save_weights("E:\\분석\\투애니원\\model_볼터치.h5")
print("Saved model to disk")


# In[59]:


# 테스트


# In[63]:


import pickle
with open('E:\\분석\\투애니원\\train_x_볼터치', 'wb') as f:
    pickle.dump(train_x, f)
with open('E:\\분석\\투애니원\\train_y_볼터치', 'wb') as f:
    pickle.dump(train_y, f)
with open('E:\\분석\\투애니원\\test_x_볼터치', 'wb') as f:
    pickle.dump(test_x, f)
with open('E:\\분석\\투애니원\\test_y_볼터치', 'wb') as f:
    pickle.dump(test_y, f)


# In[64]:


with open('E:\\분석\\투애니원\\train_x_볼터치', 'rb') as f:
    train_x = pickle.load(f) 
with open('E:\\분석\\투애니원\\train_y_볼터치', 'rb') as f:
    train_y = pickle.load(f) 
with open('E:\\분석\\투애니원\\test_x_볼터치', 'rb') as f:
    test_x = pickle.load(f) 
with open('E:\\분석\\투애니원\\test_y_볼터치', 'rb') as f:
    test_y = pickle.load(f) 


# In[65]:


from tensorflow.keras.models import model_from_json 
json_file = open("E:\\분석\\투애니원\\model_볼터치.json", "r")
loaded_model_json = json_file.read() 
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("E:\\분석\\투애니원\\model_볼터치.h5")


# In[66]:


loaded_model.summary()


# In[67]:


loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


ohe


# In[84]:


'패스' in 볼터치_o


# In[85]:


'패스' in 볼터치_x


# In[82]:


Y[10:11]


# In[83]:


np.argmax(loaded_model.predict(scaler.transform(X[10:11])))


# In[ ]:


# 임계값 50%


# In[84]:


len(test_x)


# In[85]:


pred=[]
real=[]
for i in range(len(test_x[0:5000])):
    if i % 100 ==0:
        print(i)
    if np.max(loaded_model.predict_proba(test_x[i:i+1])) >= 0.5:
        pred.append(loaded_model.predict_classes(test_x[i:i+1])[0])
        real.append(np.argmax(test_y[i]))


# In[86]:


print(len(np.where(list(np.equal(real,pred)))[0])/len(real))

print(len(real)/5000)


# In[87]:


# 임계값 60%


# In[88]:


pred=[]
real=[]
for i in range(len(test_x[0:5000])):
    if i % 100 ==0:
        print(i)
    if np.max(loaded_model.predict_proba(test_x[i:i+1])) >= 0.6:
        pred.append(loaded_model.predict_classes(test_x[i:i+1])[0])
        real.append(np.argmax(test_y[i]))


# In[89]:


print(len(np.where(list(np.equal(real,pred)))[0])/len(real))

print(len(real)/5000)


# In[90]:


# 임계값 70%


# In[91]:


pred=[]
real=[]
for i in range(len(test_x[0:5000])):
    if i % 100 ==0:
        print(i)
    if np.max(loaded_model.predict_proba(test_x[i:i+1])) >= 0.7:
        pred.append(loaded_model.predict_classes(test_x[i:i+1])[0])
        real.append(np.argmax(test_y[i]))


# In[92]:


print(len(np.where(list(np.equal(real,pred)))[0])/len(real))

print(len(real)/5000)


# In[93]:


# 임계값 80%


# In[94]:


pred=[]
real=[]
for i in range(len(test_x[0:5000])):
    if i % 100 ==0:
        print(i)
    if np.max(loaded_model.predict_proba(test_x[i:i+1])) >= 0.8:
        pred.append(loaded_model.predict_classes(test_x[i:i+1])[0])
        real.append(np.argmax(test_y[i]))


# In[95]:


print(len(np.where(list(np.equal(real,pred)))[0])/len(real))

print(len(real)/5000)


# In[96]:


# 임계값 90%


# In[97]:


pred=[]
real=[]
for i in range(len(test_x[0:5000])):
    if i % 100 ==0:
        print(i)
    if np.max(loaded_model.predict_proba(test_x[i:i+1])) >= 0.9:
        pred.append(loaded_model.predict_classes(test_x[i:i+1])[0])
        real.append(np.argmax(test_y[i]))


# In[98]:


print(len(np.where(list(np.equal(real,pred)))[0])/len(real))

print(len(real)/5000)


# In[121]:


len(np.where(pred_class==real_class)[0])/len(real_class)


# In[128]:


wrong_index=list(np.where(pred_class!=real_class)[0])


# In[131]:


len(wrong_index)


# In[136]:


wrong_index[-1]


# In[137]:


wrong_class=[]
for i in wrong_index:
    wrong_class.append(real_class[i])
    print(i)
    


# In[138]:


wrong_class


# In[143]:


wrong_class_name=[class_index[x] for x in wrong_class]


# In[147]:


total_class_name=[class_index[x] for x in real_class]


# In[153]:


total_class_name_df=pd.DataFrame(pd.Series(total_class_name).value_counts()).sort_index()


# In[155]:


total_class_name_df.columns=['테스트전체개수']


# In[156]:


total_class_name_df


# In[159]:


wrong_class_name_df=pd.DataFrame(pd.Series(wrong_class_name).value_counts()).sort_index()


# In[160]:


wrong_class_name_df.columns=['틀린개수']


# In[165]:


wrong_class_name_df.shape


# In[166]:


total_class_name_df.shape


# In[168]:


df_concat=pd.concat([total_class_name_df,wrong_class_name_df],axis=1)


# In[170]:


df_concat=df_concat.fillna(0)


# In[174]:


df_concat['틀린비율']=df_concat['틀린개수']/df_concat['테스트전체개수']*100


# In[178]:


df_concat.to_excel('E:\\분석\\투애니원\\타입별틀린비율.xlsx')


# In[182]:


new=np.concatenate((train_y, test_y), axis = 0)


# In[184]:


real_class=[np.argmax(x) for x in new]


# In[185]:


total_class_name=[class_index[x] for x in real_class]


# In[186]:


total_class_name_df=pd.DataFrame(pd.Series(total_class_name).value_counts())


# In[188]:


total_class_name_df.to_excel('E:\\분석\\투애니원\\전체클래스.xlsx')

