#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from flask import Flask , render_template , request
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[3]:


app = Flask(__name__)


# In[4]:


model = pickle.load(open('model.pkl','rb'))


# In[5]:


model


# In[6]:


@app.route('/')
def home():
    return render_template('index.html')


# In[7]:


@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [ int(x) for x in request.form.values() ]
    final_features = [ np.array(int_features)]
    prediction = model.predict(final_features)
    
    
    output = round(prediction[0], 2)
    
    return render_template('index.html',prediction_text = 'Employer Salary will be Rs {}'.format(output))


# In[9]:


if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




