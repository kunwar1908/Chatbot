import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import json
import random
import tkinter as tk
from tkinter import ttk, scrolledtext
from keras import models

# Initialize lemmatizer and load model and data
lemmatizer = WordNetLemmatizer()
model = models.load_model('C:/Users/kunwa/Python/Projects/chatbot-python-project-data-codes/chatbot_model.h5')
intents = json.loads(open('C:/Users/kunwa/Python/Projects/chatbot-python-project-data-codes/intents.json').read())
words = pickle.load(open('C:/Users/kunwa/Python/Projects/chatbot-python-project-data-codes/words.pkl','rb'))
classes = pickle.load(open('C:/Users/kunwa/Python/Projects/chatbot-python-project-data-codes/classes.pkl','rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag).reshape(1, -1)

def predict_class(sentence, model):
    # Filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(p)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def chatbot_response(msg):
    ints = predict_class(msg, model)
    if ints:
        tag = ints[0]['intent']
        list_of_intents = intents['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "I don't understand!"

# GUI setup
def send():
    msg = entry_box.get("1.0", 'end-1c').strip()
    entry_box.delete("0.0", tk.END)

    if msg != '':
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, "You: " + msg + '\n\n')
        chat_log.config(foreground="#442265", font=("Verdana", 12))
        
        res = chatbot_response(msg)
        chat_log.insert(tk.END, "Bot: " + res + '\n\n')
            
        chat_log.config(state=tk.DISABLED)
        chat_log.yview(tk.END)

base = tk.Tk()
base.title("Chatbot")
base.geometry("400x500")
base.resizable(width=False, height=False)

# Create Chat window
chat_log = scrolledtext.ScrolledText(base, bd=0, bg="white", height="8", width="50", font="Arial")
chat_log.config(state=tk.DISABLED)

# Bind scrollbar to Chat window
scrollbar = tk.Scrollbar(base, command=chat_log.yview, cursor="heart")
chat_log['yscrollcommand'] = scrollbar.set

# Create Button to send message
send_button = tk.Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                        bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                        command=send)

# Create the box to enter message
entry_box = tk.Text(base, bd=0, bg="white", width=29, height="5", font="Arial")
entry_box.bind("<Return>", lambda event: send())

# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
chat_log.place(x=6, y=6, height=386, width=370)
entry_box.place(x=6, y=401, height=90, width=265)
send_button.place(x=275, y=401, height=90)

base.mainloop()