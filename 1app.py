#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('word_completion_model_v2.h5')

# Define tokenizer and sequence length (assuming you have this from the training phase)
# You should load or recreate the tokenizer that was used in the original training
max_vocab_size = 3000
max_sequence_length = 5  # This should match your training setup

# Example of how the tokenizer might be loaded or recreated
# (You should load it from where you saved it during model training)
tokenizer = Tokenizer(num_words=max_vocab_size)
# tokenizer.fit_on_texts([...]) # Fit with the text data used in model training

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.json['text']

    # Convert input text into a sequence of tokens
    sequence = tokenizer.texts_to_sequences([input_text])[0]
    # Pad the sequence so it matches the input size expected by the model
    padded_sequence = pad_sequences([sequence], maxlen=max_sequence_length - 1, padding='pre')
    
    # Make prediction using the model
    prediction = model.predict(padded_sequence, verbose=0)
    
    # Find the word corresponding to the predicted class (highest probability)
    predicted_word_index = np.argmax(prediction)
    predicted_word = tokenizer.index_word.get(predicted_word_index, "<unknown>")
    
    # Return the predicted word in JSON format
    return jsonify({'predicted_word': predicted_word})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:





# In[ ]:




