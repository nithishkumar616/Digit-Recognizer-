#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


# In[7]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout


# In[13]:


from tensorflow.keras.datasets import fashion_mnist


# In[16]:


(x_tr,y_tr),(x_te,y_te) = fashion_mnist.load_data()
print(x_tr.shape)
print(x_te.shape)
print(y_tr.shape)
print(y_te.shape)


# In[110]:



plt.figure(figsize=(18,18))
for i in range(32):
    plt.subplot(8,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_tr[i],cmap=plt.cm.binary)
    plt.xlabel(y_tr[i])
plt.show()


# In[ ]:


print(list(set(y_tr)))


# In[33]:


x_tr = x_tr[:20000,:,:]
x_te = x_te[:3800,:,:]
y_tr = y_tr[:20000]
y_te = y_te[:3800]
print(x_tr.shape)
print(x_te.shape)
print(y_tr.shape)
print(y_te.shape)


# In[20]:


x_te[7]


# In[25]:


y_tr[:5]


# In[35]:


m1=Sequential()
m1.add(Conv2D(64,(3,3),strides=1,activation='relu',input_shape=(28,28,1)))
m1.add(MaxPooling2D(pool_size=(2,2),strides=2))

m1.add(Conv2D(64,(3,3),strides=1,activation='relu'))
m1.add(MaxPooling2D(pool_size=(2,2),strides=2))

m1.add(Flatten())
m1.add(Dense(64,activation='relu'))
m1.add(Dense(10,activation='softmax'))

m1.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])



# In[36]:


h1 = m1.fit(x_tr,y_tr,epochs=5,validation_data=(x_te,y_te))


# In[38]:


import pandas as pd

r1 = pd.DataFrame(h1.history)
r1['Epochs']=h1.epoch
r1.tail()


# In[39]:


plt.plot(r1['Epochs'],r1['loss'],label='Training loss')
plt.plot(r1['Epochs'],r1['val_loss'],label='Testing loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[40]:


plt.plot(r1['Epochs'],r1['accuracy'],label='Training accuracy')
plt.plot(r1['Epochs'],r1['val_accuracy'],label='Testing accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[41]:


ypred_m1=m1.predict(x_te)
print(ypred_m1)


# In[45]:


import numpy as np
ypred_m1=[np.argmax(i) for i in ypred_m1]
print(ypred_m1)


# In[53]:


from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_te,ypred_m1)
print(cm)
print(classification_report(y_te,ypred_m1))


# In[64]:


from tensorflow.keras.utils import to_categorical

y_tr1 = to_categorical(y_tr)
y_te1 = to_categorical(y_te)
print(y_tr1.shape)
print(y_te1.shape)


# In[65]:


y_tr1[:5]


# In[69]:


m2=Sequential()
m2.add(Conv2D(64,(3,3),strides=1,activation='relu',input_shape=(28,28,1)))
m2.add(MaxPooling2D(pool_size=(2,2),strides=2))

m2.add(Conv2D(32,(3,3),strides=1,activation='relu'))
m2.add(MaxPooling2D(pool_size=(2,2),strides=2))

m2.add(Flatten())
m2.add(Dense(64,activation='relu'))
m2.add(Dense(10,activation='softmax'))

m2.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])



# In[70]:


h2 = m2.fit(x_tr,y_tr1,epochs=5,validation_data=(x_te,y_te1))


# In[74]:


r2 = pd.DataFrame(h2.history)
r2['Epochs']=h2.epoch
r2.tail()


# In[73]:


plt.plot(r2['Epochs'],r2['accuracy'],label='Training accuracy')
plt.plot(r2['Epochs'],r2['val_accuracy'],label='Testing accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[75]:


plt.plot(r2['Epochs'],r2['loss'],label='Training loss')
plt.plot(r2['Epochs'],r2['val_loss'],label='Testing loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[77]:


ypred_m2=m2.predict(x_te)
print(ypred_m2)


# In[78]:


ypred_m2=[np.argmax(i) for i in ypred_m2]
print(ypred_m2)


# In[80]:


y_te1[:10]


# In[82]:


cm_m2 = confusion_matrix(y_te,ypred_m2)
print(cm_m2)
print(classification_report(y_te,ypred_m2))


# In[83]:


m1.save('fashion_minst.h5')


# In[84]:


from tensorflow.keras.models import load_model


# In[89]:


model=load_model('fashion_minst.h5')


# In[90]:


res= model.predict(x_te[:10])
res


# In[99]:


from tensorflow.keras.datasets import mnist


# In[102]:


(x_tr2,y_tr2),(x_te2,y_te2)=mnist.load_data()
print(x_tr2.shape)
print(x_te2.shape)
print(y_tr2.shape)
print(y_te2.shape)


# In[111]:


plt.figure(figsize=(18,18))
for i in range(32):
   plt.subplot(8,8,i+1)
   plt.xticks([])
   plt.yticks([])
   plt.imshow(x_tr2[i],cmap=plt.cm.binary)
   plt.xlabel(y_tr2[i])
plt.show()


# In[13]:


import numpy as np
import cv2
 import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('digit_rec.h5')


st.title('Digit Recognizer Streamlit App')


SIZE = 200 # size of the box where users can draw images (numbers from 1 ot 9)
mode = st.checkbox('Draw or Delete',True) 
canvas_res = st_canvas(
    fill_color='#000000',  # black
    stroke_width = 20,
    stroke_color = "#FFFFFF",  # white
    background_color = '#000000', # black
    width = SIZE,
    height = SIZE,
    drawing_mode = 'freedraw' if mode else 'transform',
    key = 'canvas')

if canvas_res.image_data is not None:
    img = cv2.resize(canvas_res.image_data.astype('uint8'),(28,28))
    rescaled = cv2.resize(img,(SIZE,SIZE),interpolation=cv2.INTER_NEAREST)
    st.write('This image will be used as Model imput')
    st.image(rescaled)


if st.button('Predict'):
    test_x = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    res = model.predict(test_x.reshape(1,28,28,1))
    st.write(f'Result is {np.argmax(res[0])}')
    st.write(res)
    st.bar_chart(res[0])


# In[ ]:





# In[ ]:




