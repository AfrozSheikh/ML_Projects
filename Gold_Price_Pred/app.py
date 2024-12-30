from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
with open("./model/file_name.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form
        spx = float(request.form['SPX'])
        uso = float(request.form['USO'])
        slv = float(request.form['SLV'])
        eur_usd = float(request.form['EUR/USD'])

        # Create input array for the model
        input_features = np.array([[spx, uso, slv, eur_usd]])
        
        # Make prediction
        predicted_gld = model.predict(input_features)[0]

        return jsonify({'prediction': f'Predicted GLD: {predicted_gld:.2f}'})
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'})

if __name__ == "__main__":
    app.run(debug=True)
