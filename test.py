import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import pickle
from sklearn.preprocessing import LabelEncoder
import random

# Load thông tin từ điển và nhãn lớp
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Load mô hình đã huấn luyện
model = keras.models.load_model('chatbot_model.h5')

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

def get_response(input_text):
    # Tiền xử lý câu đầu vào
    input_text = [input_text]
    input_sequences = tokenizer.texts_to_sequences(input_text)
    input_sequences = keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
    
    # Dự đoán lớp của câu đầu vào
    predicted_class = model.predict(input_sequences)
    predicted_class = np.argmax(predicted_class)
    
    # Lấy ra câu trả lời tương ứng với lớp dự đoán
    response_list = responses[label_encoder.inverse_transform([predicted_class])[0]]
     # Chọn một câu trả lời ngẫu nhiên từ danh sách
    response = random.choice(response_list)
    return response

while True:
    user_input = input("You: ")
    if user_input.lower() == 'Bye':
        print("Tạm biệt, hẹn gặp lại.")
        break

    response = get_response(user_input)
    print("Bot:", response)