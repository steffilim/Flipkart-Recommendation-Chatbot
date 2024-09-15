from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send-message', methods=['POST'])
def send_message():
    data = request.get_json()
    user_message = data['message']
    # Simulated response, replace with actual logic later
    return jsonify({'response': f'Echo: {user_message}'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
