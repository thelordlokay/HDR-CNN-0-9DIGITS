from model.network import Net
from flask import Flask, jsonify, render_template, request, redirect, url_for, session, send_file
from flask_mysqldb import MySQL
from preprocessing import *
import secrets
import os
import re

app = Flask(__name__)

# Generate a random 32-byte secret key
app.secret_key = secrets.token_hex(32)

# MySQL configurations
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'database'

mysql = MySQL(app)

# Helper function to check if user is logged in
def is_logged_in():
    return 'username' in session

# Helper function to render templates based on login status
def render_template_with_login(template_name):
    if is_logged_in():
        return render_template(template_name)
    return redirect(url_for('login'))

@app.route('/')
def login_redirect():
    if is_logged_in():
        return redirect(url_for('main'))
    return redirect(url_for('login'))  # Redirect to login page if not logged in


@app.route('/main')
def main():
    if not is_logged_in():
        return redirect(url_for('login'))
    return render_template('main.html')

@app.route('/index.html', endpoint='index_page')
def index():
    return render_template_with_login('index.html')

@app.route('/digit_prediction', methods=['POST'])
def digit_prediction():
    if request.method == "POST":
        img = preprocess(request.get_json())
        net = Net()
        digit, probability = net.predict_with_pretrained_weights(img, 'pretrained_weights.pkl')
        session['probability'] = round(probability, 2)  # Save probability in session with 2 decimal places
        return jsonify({"digit": int(digit), "probability": round(probability, 2)})

@app.route('/report')
def report():
    probability = session.get('probability')
    return render_template('report.html', probability=probability)

@app.route('/training_log.txt')
def serve_training_log():
    training_log_path = os.path.join(os.getcwd(), 'training_log.txt')
    return send_file(training_log_path)

@app.route('/display')
def display():
    return render_template('display.html')

@app.route("/contact", methods=["POST"])
def contact():
    name, email, subject, message = request.form["name"], request.form["email"], request.form["subject"], request.form["message"]
    send_email(name, email, subject, message)
    return redirect(url_for('main'))

@app.route('/home')
def home():
    return render_template('home.html', username=session['username']) if is_logged_in() else render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username, pwd = request.form['username'], request.form['password']
        if not username or not pwd:
            return render_template('login.html', error='Username and password are required.')
        cursor = mysql.connection.cursor()
        cursor.execute(f"SELECT username,password FROM user WHERE username = '{username}'")
        record = cursor.fetchone()
        cursor.close()
        if record and pwd == record[1]:
            session['username'] = record[0]
            return redirect(url_for('main'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username, pwd, email = request.form['username'], request.form['password'], request.form['email']
        if not username or not pwd or not email:
            return render_template('register.html', error='All fields are required.')
        
        # Password strength check
        if not is_strong_password(pwd):
            return render_template('register.html', error='Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one digit, and one special character.')

        cursor = mysql.connection.cursor()
        cursor.execute(f"SELECT * FROM user WHERE username = '{username}'")
        record = cursor.fetchone()
        if record:
            error_msg = 'Username already exists. Please choose a different username.'
        else:
            cursor.execute(f"INSERT INTO user (username,password, email) VALUES ('{username}','{pwd}','{email}') ")
            mysql.connection.commit()
            error_msg = None
        cursor.close()  # Close cursor after processing
        if error_msg:
            return render_template('register.html', error=error_msg)
        else:
            return redirect(url_for('login'))
    return render_template('register.html', error='')

def is_strong_password(password):
    # Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one digit, and one special character
    if len(password) < 8:
        return False
    if not re.search("[a-z]", password):
        return False
    if not re.search("[A-Z]", password):
        return False
    if not re.search("[0-9]", password):
        return False
    if not re.search("[!@#$%^&*()-_+=]", password):
        return False
    return True


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)




