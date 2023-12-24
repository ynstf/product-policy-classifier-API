from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from tensorflow import keras
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import pickle
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt')  # Download the punkt tokenizer if you haven't already
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from detect import detect_language
from translate import translate_text

text_to_translate = "Sexual products"
language = detect_language(text_to_translate)
translated_text = translate_text(text_to_translate, source_lang=language , target_lang='en')

app = FastAPI()

# Load the pre-trained model
version = 6
model_folder = f"model_v{version}"
model = keras.models.load_model(f'{model_folder}/model.h5')

# Load tokenizer configuration from the file
with open(f'{model_folder}/tokenizer_config.json', 'r') as json_file:
    tokenizer_config_str = json_file.read()
# Create a tokenizer instance using tokenizer_from_json
tokenizer = tokenizer_from_json(tokenizer_config_str)

#load trainig data
# Specify the file path where you saved the data
pickle_file_path = f'{model_folder}/training_data.pkl'
# Load the training_data dictionary from the Pickle file
with open(pickle_file_path, 'rb') as pickle_file:
    loaded_training_data = pickle.load(pickle_file)
# Access the loaded data
max_words = loaded_training_data['max_words']
max_sequence = loaded_training_data['max_sequence']
legend = loaded_training_data['legend']
labels_legend_inverted = loaded_training_data['labels_legend_inverted']


# define the stem function
def stem_text(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    stemmed_text = ' '.join(stemmed_tokens)
    return stemmed_text


# define the pridection function
def predict(text_str, max_sequence=max_sequence, tokenizer=None, model=None, labels_legend_inverted=None):
    if not tokenizer or not model or not labels_legend_inverted:
        return None
    
    #detect language
    text_to_translate = text_str
    language = detect_language(text_to_translate)

    #translate text
    translated_text = translate_text(text_to_translate, source_lang=language , target_lang='en')
    print(translated_text)
    #stemming the input text
    text_str = stem_text(translated_text)
    # Tokenize the input text
    sequences = tokenizer.texts_to_sequences([text_str])
    # Pad the sequence
    x_input = pad_sequences(sequences, maxlen=max_sequence)
    # Predict using the model
    y_output = model.predict(x_input,verbose=0)
    # Assuming you want to get the label with the highest probability
    top_y_index = np.argmax(y_output, axis=-1)[0]
    preds = y_output[0][top_y_index]
    labeled_preds = {labels_legend_inverted[str(top_y_index)]: float(preds)}
    return labels_legend_inverted[str(top_y_index)],labeled_preds




class InputData(BaseModel):
    data: str

@app.get("/")
def welcome():
    return JSONResponse("welcome")

@app.post("/predict")
def predicting(data: InputData):
    data = data.data
    print(data)
    prediction, x = predict(str(data), tokenizer=tokenizer, model=model, labels_legend_inverted=labels_legend_inverted)
    print(prediction, x)
    return JSONResponse(content={"prediction": prediction,"precision":x[prediction]})