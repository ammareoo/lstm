#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import requests

st.title("Word Completion using LSTM")

# Get user input
user_input = st.text_input("Type a sentence:")

# Make API request if there is user input
if user_input:
    response = requests.post('http://localhost:5000/predict', json={'text': user_input})
    predicted_word = response.json().get('predicted_word', 'Error predicting word')
    st.write(f"Predicted next word: {predicted_word}")


# In[ ]:





# In[ ]:




