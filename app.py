from flask import Flask, render_template, request, session
import pickle
from models.HM_Task_2 import heuristic_model
import numpy as np
app = Flask(__name__) #Creating Flask Instance
#Loading models
RFC_model = pickle.load(open(r'models/saved_models/RFC_model.pkl', 'rb'))
KNN_model = pickle.load(open(r'models/saved_models/KNN_model.pkl', 'rb'))
NN_model = pickle.load(open(r'models/saved_models/NN_model.pkl', 'rb'))


def predict(model, x1, x2):  #Predict method For NN model we need to modify it
    if model == NN_model:
        return np.argmax(model.predict([[x1, x2]]))
    else:
        return model.predict([[x1, x2]])[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict_route():     #I defined how to get values and and how to use them
    model_choice = request.form['model'] #in diffrent cases
    x1 = float(request.form['x1'])
    x2 = float(request.form['x2'])


    if model_choice == 'Heuristic_Classifier':
        result = heuristic_model(x1, x2)
    elif model_choice == 'Random_Forest_Classifier':
        result = predict(RFC_model, x1, x2)
    elif model_choice == 'KNN_Classifier':
        result = predict(KNN_model, x1, x2)
    else:
        result = predict(NN_model, x1, x2)

    return render_template('index.html', result= f'Cover type: {result}')

if __name__ == '__main__':
    app.run(debug=True)
