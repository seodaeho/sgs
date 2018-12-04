
# coding: utf-8

# In[31]:


import pymysql
import datetime
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# In[52]:


conn=pymysql.connect(database="epts",user="epts",password="SGSepts123!",host="211.254.217.49", port=3306,charset='utf8')


# In[53]:


cursor = conn.cursor()


# In[54]:


cursor.execute("SHOW TABLES")


# In[55]:


tables = cursor.fetchall()


# In[56]:


tables


# In[57]:


sql = "select * from device_data"
cursor.execute(sql)


# In[58]:


num_fields = len(cursor.description)
field_names = [i[0] for i in cursor.description]


# 필드네임

# In[59]:


rows = cursor.fetchall()


# In[60]:


len(rows)


# 데이터 전처리

# In[61]:


Y=[]
X=[]
for i in range(len(rows)):
    Y.append('_'.join(rows[i][5:9]))
    X.append(rows[i][43:52])


# In[62]:


len(np.unique(Y))


# In[63]:


Y = pd.DataFrame(Y,columns=['class'])
X = pd.DataFrame(X)


# In[64]:


Data_pre=pd.concat([Y,X],axis = 1)


# In[65]:


Data_pre.iloc[17990,1] ==""


# In[66]:


filter = Data_pre[0] != ""
Data_pre_new = Data_pre[filter]


# In[67]:


Data_pre_new=Data_pre_new.reset_index(drop=True)


# In[91]:


len(Data_pre_new)


# In[68]:


Data_pre_new.head()


# 전처리후 엑스 와이 나누기

# In[69]:


Y2=Data_pre_new['class']
X2=Data_pre_new.iloc[:,1:10]


# In[70]:


np.unique(Y2)


# In[71]:


len(np.unique(Y2))


# 모델만들기

# In[72]:


from sklearn.preprocessing import MinMaxScaler


# In[73]:


Y2_class=pd.get_dummies(Y2,prefix='class_')


# In[74]:


# 뉴럴넷 모델 디자인
model = Sequential()
model.add(Dense(1000, input_dim=(X2.shape[1]), activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(len(np.unique(Y2)), activation='softmax'))


# In[75]:


# 0~1 정규화 시행
scaler=MinMaxScaler(feature_range=(0,1))
scaler.fit(X2)


# In[76]:


dataset_x = scaler.transform(X2)


# In[77]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[78]:


msk = np.random.rand(len(dataset_x)) < 0.7


# In[79]:


train_x = dataset_x[msk]
train_y= Y2_class[msk]
test_x = dataset_x[~msk]
test_y = Y2_class[~msk]


# In[104]:


# 500회 훈련, 배치사이즈 5000 훈련주기는 더 늘리면 좋을듯. 지금은 시간상 150번만함.
model.fit(train_x, train_y, epochs=500, batch_size=5000, verbose=2)


# In[105]:


predicted_classes = model.predict(test_x)


# In[106]:


predicted_classes = np.argmax(np.round(predicted_classes),axis=1)


# In[107]:


predicted_classes


# In[108]:


real=[]


# In[109]:


for i in range(len(test_y)):
    real.append(int(np.where(test_y.iloc[i,:] == 1)[0]))


# In[110]:


test_eval = model.evaluate(test_x, test_y, verbose=0)


# In[111]:


test_eval[1]

