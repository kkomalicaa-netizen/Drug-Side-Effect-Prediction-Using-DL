import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv("../dataset/drug_data.csv")

text = df["description"]
structured = df[["age","gender","dosage"]].values
labels = df.iloc[:,4:].values

# Text processing
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(text)
seq = tokenizer.texts_to_sequences(text)
text_pad = pad_sequences(seq, maxlen=20)

# Scale structured features
scaler = StandardScaler()
structured_scaled = scaler.fit_transform(structured)

X_train_text, X_test_text, X_train_struct, X_test_struct, y_train, y_test = train_test_split(
    text_pad, structured_scaled, labels, test_size=0.2)

# TEXT INPUT
text_input = Input(shape=(20,))
x = Embedding(2000, 64)(text_input)
x = LSTM(64)(x)

# STRUCTURED INPUT
struct_input = Input(shape=(3,))
y = Dense(32, activation='relu')(struct_input)

# MERGE
combined = Concatenate()([x,y])
z = Dense(64, activation='relu')(combined)
output = Dense(10, activation='sigmoid')(z)

model = Model(inputs=[text_input, struct_input], outputs=output)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit([X_train_text, X_train_struct], y_train, epochs=10, validation_split=0.2)

loss, acc = model.evaluate([X_test_text, X_test_struct], y_test)
print("Model Accuracy:", acc)

os.makedirs(".", exist_ok=True)
model.save("model.h5")

with open("tokenizer.pkl","wb") as f:
    pickle.dump(tokenizer,f)

with open("scaler.pkl","wb") as f:
    pickle.dump(scaler,f)
