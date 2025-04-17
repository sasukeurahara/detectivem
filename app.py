import requests
import json
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, logout_user, login_required, current_user
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify


app = Flask(__name__)

url = "http://localhost:11434/api/generate"
headers = {
    'Content-Type': 'application/json'
}

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///meow.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'  # Add a secret key for session management

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Redirect to login page if not authenticated

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(80), nullable=False)
    pno = db.Column(db.String(80), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.before_request
def create_tables():
    db.create_all()

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # TO DO: Verify user credentials and log in
        # For now, just assume the login is successful
        login_user(username)
        flash("Login successful!")

        next_page = request.args.get('next')
        if next_page:
            return redirect(next_page)
        else:
            return redirect(url_for("home"))  # Redirect to home page after successful login
    return render_template('login.html')

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        pno = request.form['tel']

        if User.query.filter_by(username=username).first():
            flash("Username already exists!")
        else:
            new_user = User(username=username, password=password, email=email, pno=pno)
            db.session.add(new_user)
            db.session.commit()
            flash("User successfully registered!")
            return redirect(url_for("login"))
    
    return render_template('signup.html')

@app.route("/home", methods=["GET", "POST"])
def home():
    return render_template('homepage.html')

@app.route("/logout", methods=["GET", "POST"])
def logout():
    logout_user()
    flash("You have been logged out.")
    return redirect(url_for("login"))

history = []

@app.route('/aichat', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        history.append(prompt)
        final_prompt = '\n'.join(history)

        data = {
            'model': 'dectm',
            'prompt': final_prompt,
            'stream': False
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            data = response.json()
            actual = data['response']
            history.append(actual)
        else:
            actual = f"Error: {response.text}"

        return render_template('Chatbot.html', response=actual, history=history)
    
    return render_template('Chatbot.html', response='', history=history)

model = load_model('modelfile/m500.h5')

def file_to_binary(file):
    return file.read()

def binary_to_image(data, image_size=(500, 500)):
    num_pixels = (len(data) + 2) // 3
    square_size = int(np.ceil(np.sqrt(num_pixels)))
    padded_data = np.zeros(square_size * square_size * 3, dtype=np.uint8)
    padded_data[:len(data)] = np.frombuffer(data, dtype=np.uint8)
    image_array = padded_data.reshape((square_size, square_size, 3))
    image = Image.fromarray(image_array)
    image = image.resize(image_size, Image.Resampling.LANCZOS)
    return image

def preprocess_and_predict(file):
    binary_data = file_to_binary(file)
    image = binary_to_image(binary_data)
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    resized_image = cv2.resize(opencv_image, (400, 400))
    image_array = np.array([resized_image]) / 255.0
    predictions = model.predict(image_array)
    predictions_df = pd.DataFrame(predictions)
    return predictions_df

def predict(file):
    predictions = preprocess_and_predict(file)
    result = "Safe" if predictions.iloc[0][0] < 1.0 else "Virus"
    return result

@app.route('/predict', methods=['POST'])
def predict_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        result = predict(file)
        return jsonify({"result": result})
    else:
        return jsonify({"error": "File processing error"}), 500


if __name__ == '__main__':
    app.run(debug=True)