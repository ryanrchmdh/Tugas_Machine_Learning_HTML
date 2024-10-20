from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model/hasil_pelatihan_model.pkl", "rb") as model_file:
    ml_model = joblib.load(model_file)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    print("Prediction started")
    
    if request.method == 'POST':
        try:
            # Get the input values from the form
            RnD_Spend = float(request.form['RnD_Spend'])
            Admin_Spend = float(request.form['Admin_Spend'])
            Market_Spend = float(request.form['Market_Spend'])
            
            # Prepare the input data for the model
            pred_args = [RnD_Spend, Admin_Spend, Market_Spend]
            pred_args_arr = np.array(pred_args).reshape(1, -1)
            
            # Make the prediction
            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction), 2)
        
        except ValueError:
            return "Please check if the values are entered correctly"
        
        return render_template('predict.html', prediction=model_prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
