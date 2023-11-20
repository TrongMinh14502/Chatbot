import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu từ tệp JSON
with open('Data.json', encoding='utf-8') as file:
    data = json.load(file)

# Tạo danh sách các mẫu câu đầu vào và nhãn lớp
patterns = []
labels = []
responses = {}

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        labels.append(intent['tag'])
    responses[intent['tag']] = intent['responses']

# Xây dựng bộ từ điển từ các mẫu câu đầu vào
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(patterns)
total_words = len(tokenizer.word_index) + 1

# Chuyển các mẫu câu đầu vào thành dạng số nguyên
input_sequences = tokenizer.texts_to_sequences(patterns)

# Padding để có độ dài cố định cho các mẫu câu đầu vào
max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')

# Chuyển nhãn lớp thành dạng số nguyên
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Xây dựng mô hình mạng nơron
model = keras.Sequential([
    keras.layers.Embedding(total_words, 100, input_length=max_sequence_length),
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(data['intents']), activation='softmax')
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(input_sequences, encoded_labels, epochs=80)

# Lưu mô hình đã huấn luyện
model.save('chatbot_model.h5')

# Lưu thông tin về từ điển và nhãn lớp
import pickle

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('label_encoder.pickle', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)