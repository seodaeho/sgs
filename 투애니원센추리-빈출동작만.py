
# coding: utf-8

# In[4]:


from tensorflow.keras.models import model_from_json
import pickle
import time
import pymysql
import datetime
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import ast


# In[5]:


def flatten(l): 
    flatList = [] 
    for elem in l: 
        if type(elem) == list: 
            for e in elem: 
                flatList.append(e) 
        else: 
            flatList.append(elem) 
    return flatList



# In[5]:


conn=pymysql.connect(database="epts",user="epts",password="SGSepts123!",host="125.129.225.44", port=3306)
cursor = conn.cursor()


# In[20]:


conn.set_charset('utf8')
#sql = "select * from device_data_temp2 where datetime>='2019-01-01'"
sql = "select * from device_data_temp2 where datetime>='2019-03-01'"
cursor.execute(sql)
rows = cursor.fetchall()


# In[24]:


import pickle


# In[ ]:


from sklearn.externals import joblib 

file_name = 'E:\\분석\\투애니원\\rows_동작만.pkl' 
joblib.dump(rows, file_name)


# In[39]:


# 분석부분


# In[25]:


from sklearn.externals import joblib 
file_name = 'E:\\분석\\투애니원\\rows_동작만.pkl' 
rows = joblib.load(file_name)


# In[ ]:


import pymysql
conn=pymysql.connect(database="epts",user="epts",password="SGSepts123!",host="125.129.225.44", port=3306)
cursor = conn.cursor()
conn.set_charset('utf8')
sql = "select * from device_data_temp2 limit 1"
cursor.execute(sql)
colnames = [desc[0] for desc in cursor.description]


# In[ ]:


import pandas as pd


# In[ ]:


df=pd.DataFrame(rows,columns=colnames)


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df=df.iloc[:,[8,44,45,46]]


# In[ ]:


df=df.dropna().reset_index(drop=True)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[6]:


import numpy as np
import ast
from sklearn.preprocessing import MinMaxScaler
import json
# 중첩 리스트 하나의 리스트로 변환하는 함수
def flatten(l): 
    flatList = [] 
    for elem in l: 
        if type(elem) == list: 
            for e in elem: 
                flatList.append(e) 
        else: 
            flatList.append(elem) 
    return flatList



# In[ ]:


Y=[]
X=[]
for i in range(0,len(rows)):
    try:
        temp=[]
        temp=temp+json.loads(df['before_items'][i])['before10']
        temp=temp+json.loads(df['before_items'][i])['before9']
        temp=temp+json.loads(df['before_items'][i])['before8']
        temp=temp+json.loads(df['before_items'][i])['before7']
        temp=temp+json.loads(df['before_items'][i])['before6']
        temp=temp+json.loads(df['before_items'][i])['before5']
        temp=temp+json.loads(df['before_items'][i])['before4']
        temp=temp+json.loads(df['before_items'][i])['before3']
        temp=temp+json.loads(df['before_items'][i])['before2']
        
        temp=temp+json.loads(df['before_items'][i])['before1']
        temp=temp+(ast.literal_eval(df['point'][i]))
        temp=temp+(json.loads(df['after_items'][i])['after1'])
        temp=temp+(json.loads(df['after_items'][i])['after2'])
        temp=temp+(json.loads(df['after_items'][i])['after3'])
        temp=temp+(json.loads(df['after_items'][i])['after4'])
        temp=temp+(json.loads(df['after_items'][i])['after5'])
        temp=temp+(json.loads(df['after_items'][i])['after6'])
        temp=temp+(json.loads(df['after_items'][i])['after7'])
        temp=temp+(json.loads(df['after_items'][i])['after8'])
        temp=temp+(json.loads(df['after_items'][i])['after9'])
        temp=temp+(json.loads(df['after_items'][i])['after10'])
        X.append(temp)

        Y.append(df['type4'][i])
    except Exception as e:
        print(e)
        continue



# In[ ]:


Y = pd.DataFrame(Y,columns=['class'])
X = pd.DataFrame(X)


# In[ ]:


file_name = 'E:\\분석\\투애니원\\X.pkl' 
joblib.dump(X, file_name)
file_name = 'E:\\분석\\투애니원\\Y.pkl' 
joblib.dump(Y, file_name)


# In[ ]:


# 여기부터 실행


# In[170]:


from sklearn.externals import joblib 
file_name = 'E:\\분석\\투애니원\\X.pkl' 
X = joblib.load(file_name)
file_name = 'E:\\분석\\투애니원\\Y.pkl' 
Y = joblib.load(file_name)


# In[171]:


X=X.reset_index(drop=True)
Y=Y.reset_index(drop=True)


# In[172]:


len(X)


# In[173]:


X.head()


# In[174]:


X.tail()


# In[175]:


Y.tail()


# In[176]:


X_Y=pd.concat([X,Y], axis=1)


# In[177]:


X_Y.head(10)


# In[178]:


X_Y=X_Y.dropna().reset_index(drop=True)


# In[179]:


X_Y.head()


# In[180]:


freq_class=['원터치패스',
'땅볼트래핑',
'턴',
'이동드리블',
'볼터치',
'스텝오버',
'스프린트',
'킥',
'헤딩슈팅',
'논스톱패스',
'이동트래핑',
'러닝',
'턴트래핑',
'팬텀드리블',
'크루이프턴',
'맥기디스핀',
'레이트크로싱',
'원터치슈팅',
'마르세유턴',
'롱킥',
'컷백',
'플리플랩',
'워킹',
'직진드리블']


# In[181]:


X_Y=X_Y[X_Y["class"].isin(freq_class)].reset_index(drop=True)


# In[182]:


X_Y=X_Y.groupby('class').apply(lambda x: x.sample(10000))



# In[183]:


np.unique(X_Y['class'])


# In[184]:


pd.Series(X_Y['class']).value_counts()


# In[185]:


X_Y=X_Y.reset_index(drop=True)


# In[186]:


X_Y=X_Y.sample(frac=1).reset_index(drop=True)


# In[187]:


len(np.unique(X_Y['class']))


# In[188]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore', sparse = False)


# In[189]:


np.unique(X_Y[['class']])


# In[190]:


ohe.fit(X_Y[['class']])
cat_ohe = ohe.transform(X_Y[['class']])


# In[191]:


np.unique(cat_ohe)


# In[192]:


X_Y.shape


# In[193]:


X=X_Y.iloc[:,:-1]


# In[194]:


X.head()


# In[195]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json


# In[196]:


import tensorflow as tf


# In[197]:


leaky_relu = tf.nn.leaky_relu


# In[198]:


model = Sequential()
model.add(Dense(1000, input_dim=(X.shape[1]), activation='relu'))
model.add(Dense(800, activation=leaky_relu))
model.add(Dense(350, activation=leaky_relu))
model.add(Dense(250, activation=leaky_relu))
model.add(Dense(150, activation=leaky_relu))
model.add(Dense(150, activation=leaky_relu))
model.add(Dense(100, activation=leaky_relu))
model.add(Dense(len(np.unique(X_Y[['class']])), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


# In[199]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# In[200]:


scaler=MinMaxScaler()
scaler.fit(X)
dataset_x = scaler.transform(X)


# In[201]:


import pickle
with open('E:\\분석\\투애니원\\scaler_freq_class', 'wb') as f:
    pickle.dump(scaler, f)
with open('E:\\분석\\투애니원\\ohe_freq_class', 'wb') as f:
    pickle.dump(ohe, f)


# In[202]:


msk = np.random.rand(len(dataset_x)) < 0.8
train_x = dataset_x[msk]
train_y= cat_ohe[msk]
test_x = dataset_x[~msk]
test_y = cat_ohe[~msk]


# In[203]:


np.unique(train_y)


# In[204]:


early_stopping = EarlyStopping(monitor='val_loss', patience=50)


# In[205]:


model.fit(train_x, train_y, epochs=1500, batch_size=250, validation_data=(test_x, test_y),verbose=1, callbacks=[early_stopping])


# In[206]:


#model_json = model.to_json()
#with open("E:\\분석\\투애니원\\model_freq_class.json", "w") as json_file : 
#    json_file.write(model_json)
#model.save_weights("E:\\분석\\투애니원\\model_freq_class.h5")


# In[207]:


model.save('E:\\분석\\투애니원\\model_freq_class.h5')
print("Saved model to disk")


# In[208]:


import pickle
with open('E:\\분석\\투애니원\\train_x_freq_class', 'wb') as f:
    pickle.dump(train_x, f)
with open('E:\\분석\\투애니원\\train_y_freq_class', 'wb') as f:
    pickle.dump(train_y, f)
with open('E:\\분석\\투애니원\\test_x_freq_class', 'wb') as f:
    pickle.dump(test_x, f)
with open('E:\\분석\\투애니원\\test_y_freq_class', 'wb') as f:
    pickle.dump(test_y, f)


# In[209]:


with open('E:\\분석\\투애니원\\train_x_freq_class', 'rb') as f:
    train_x = pickle.load(f) 
with open('E:\\분석\\투애니원\\train_y_freq_class', 'rb') as f:
    train_y = pickle.load(f) 
with open('E:\\분석\\투애니원\\test_x_freq_class', 'rb') as f:
    test_x = pickle.load(f) 
with open('E:\\분석\\투애니원\\test_y_freq_class', 'rb') as f:
    test_y = pickle.load(f) 


# In[210]:


from tensorflow.keras.models import model_from_json 
#json_file = open("E:\\분석\\투애니원\\model_freq_class.json", "r")
#loaded_model_json = json_file.read() 
#json_file.close()
loaded_model = tf.keras.models.load_model('E:\\분석\\투애니원\\model_freq_class.h5', custom_objects={'leaky_relu':leaky_relu})
#loaded_model.load_weights("")


# In[211]:


loaded_model.summary()


# In[212]:


loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[213]:


# 임계값 50%


# In[214]:


test_y[14120]


# In[215]:


pred=[]
real=[]
for i in range(1000):
    if i % 100 ==0:
        print(i)
    if np.max(loaded_model.predict_proba(test_x[i:i+1])) >= 0.5:
        pred.append(loaded_model.predict_classes(test_x[i:i+1])[0])
        real.append(np.argmax(test_y[i]))


# In[216]:


print(len(np.where(list(np.equal(real,pred)))[0])/len(real))

print(len(real)/1000)


# In[217]:


# 임계값 60%


# In[218]:


pred=[]
real=[]
for i in range(1000):
    if i % 100 ==0:
        print(i)
    if np.max(loaded_model.predict_proba(test_x[i:i+1])) >= 0.6:
        pred.append(loaded_model.predict_classes(test_x[i:i+1])[0])
        real.append(np.argmax(test_y[i]))


# In[219]:


print(len(np.where(list(np.equal(real,pred)))[0])/len(real))

print(len(real)/1000)


# In[220]:


# 임계값 70%


# In[221]:


pred=[]
real=[]
for i in range(1000):
    if i % 100 ==0:
        print(i)
    if np.max(loaded_model.predict_proba(test_x[i:i+1])) >= 0.7:
        pred.append(loaded_model.predict_classes(test_x[i:i+1])[0])
        real.append(np.argmax(test_y[i]))


# In[222]:


print(len(np.where(list(np.equal(real,pred)))[0])/len(real))

print(len(real)/1000)


# In[223]:


# 임계값 80%


# In[224]:


pred=[]
real=[]
for i in range(1000):
    if i % 100 ==0:
        print(i)
    if np.max(loaded_model.predict_proba(test_x[i:i+1])) >= 0.8:
        pred.append(loaded_model.predict_classes(test_x[i:i+1])[0])
        real.append(np.argmax(test_y[i]))


# In[225]:


print(len(np.where(list(np.equal(real,pred)))[0])/len(real))

print(len(real)/1000)


# In[226]:


# 임계값 90%


# In[227]:


pred=[]
real=[]
for i in range(1000):
    if i % 100 ==0:
        print(i)
    if np.max(loaded_model.predict_proba(test_x[i:i+1])) >= 0.9:
        pred.append(loaded_model.predict_classes(test_x[i:i+1])[0])
        real.append(np.argmax(test_y[i]))


# In[228]:


print(len(np.where(list(np.equal(real,pred)))[0])/len(real))

print(len(real)/1000)


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

