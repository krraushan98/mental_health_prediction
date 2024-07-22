from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model, label dictionary, feature names, and age scaler
model = joblib.load('adaboost_model.pkl')
label_dict = joblib.load('label_dict.pkl')
feature_names = joblib.load('feature_names.pkl')
age_scaler = joblib.load('age_scaler.pkl')

def preprocess_input(user_input):
    processed_input = []
    for feature in feature_names:
        value = user_input.get(feature)
        if feature == 'Age':
            # Scale the age input
            scaled_age = age_scaler.transform([[float(value)]])[0][0]
            processed_input.append(scaled_age)
        else:
            label_key = f'label_{feature}'
            if label_key in label_dict:
                # For categorical features, map the string input to its encoded value
                encoded_value = label_dict[label_key].index(value)
                processed_input.append(encoded_value)
            else:
                # For other numerical features, just convert to float
                processed_input.append(float(value))
    return np.array(processed_input).reshape(1, -1)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        user_input = {feature: request.form.get(feature) for feature in feature_names}
        processed_input = preprocess_input(user_input)
        prediction = model.predict(processed_input)[0]
        
        result = "might benefit from mental health treatment" if prediction == 1 else "is less likely to need mental health treatment"
        
        return render_template('result.html', prediction=prediction, result=result)
    
    return render_template('input.html', features=feature_names, label_dict=label_dict)

if __name__ == '__main__':
    app.run(debug=True)