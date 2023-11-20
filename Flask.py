# Import các thư viện cần thiết
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pickle
import json
import random

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

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

# Nạp mô hình đã huấn luyện
model = tf.keras.models.load_model('chatbot_model.h5')

# Nạp thông tin từ điển và nhãn lớp
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Hàm để dự đoán câu trả lời từ chatbot
def get_bot_response(user_input):
    # Tiền xử lý dữ liệu đầu vào
    user_input = [user_input]
    user_input = tokenizer.texts_to_sequences(user_input)
    user_input = tf.keras.preprocessing.sequence.pad_sequences(user_input, maxlen=9, padding='post')

    # Dự đoán nhãn lớp
    prediction = model.predict(user_input)

    # Lấy nhãn lớp có xác suất cao nhất
    predicted_label = label_encoder.inverse_transform([tf.argmax(prediction, axis=1).numpy()[0]])

    # Tìm câu trả lời dựa trên nhãn lớp
    matching_responses = responses.get(predicted_label[0], [])

    if matching_responses:
        response = random.choice(matching_responses)
    else:
        response = "Tôi không hiểu bạn đang nói gì."

    return response

# Định nghĩa route cho trang chính
@app.route('/')
def index():
    return render_template('index.html')

# Định nghĩa route cho giao tiếp với chatbot
@app.route('/get_response', methods=['POST'])
def chatbot_response():
    user_input = request.form['user_input']
    bot_response = get_bot_response(user_input)
    return jsonify({'bot_response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)