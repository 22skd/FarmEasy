from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

# importing model
model = pickle.load(open('Crop-Recommendation-System-Using-Machine-Learning\model.pkl','rb'))
sc = pickle.load(open('Crop-Recommendation-System-Using-Machine-Learning\standscaler.pkl','rb'))
ms = pickle.load(open('Crop-Recommendation-System-Using-Machine-Learning\minmaxscaler.pkl','rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                         8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                         14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                         19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}
    crop=None
    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        # Check if any input value is negative
        if any(x < 0 for x in [N, P, K, temp, humidity, ph, rainfall]):
            result = "Sorry, input values cannot be negative. Please enter valid data."
        else:
            feature_list = [N, P, K, temp, humidity, ph, rainfall]
            single_pred = np.array(feature_list).reshape(1, -1)

            scaled_features = ms.transform(single_pred)
            final_features = sc.transform(scaled_features)
            prediction = model.predict(final_features)

           
            if prediction[0] in crop_dict:
                crop = crop_dict[prediction[0]]
                result = "{} is the best crop to be cultivated right there".format(crop)
            else:
                result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    except ValueError:
        result = "Invalid input. Please enter numeric values for all fields."

    return render_template('index.html', result=result,crop_dict=crop_dict,crop=crop)





# python main
if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')