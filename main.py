from flask import Flask, render_template, request, jsonify
from clydeai import ClydeAI

app = Flask(__name__)
clyde = ClydeAI()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    messages = request.json['messages']

    # Call ClydeAI API
    response = clyde.ChatCompletion().create(
        model="clyde-1.1-mini",
        messages=messages
    )

    return jsonify({'response': response.content})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
