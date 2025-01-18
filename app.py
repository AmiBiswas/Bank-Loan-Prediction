from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def Loan_predict():
    a = request.form.get("Credit Score")
    b = request.form.get("Annual Income")
    c = request.form.get("Years in current job")
    d = request.form.get("Home Ownership")
    e = request.form.get("Monthly Debt")
    f = request.form.get("Years of Credit History")
    g = request.form.get("Number of Open Accounts")
    h = request.form.get("Current Credit Balance")
    i = request.form.get("Maximum Open Credit")

    # Ensure all fields are converted to the correct type
    a = float(a)
    b = float(b)
    c = float(c)
    d = int(d)
    e = float(e)
    f = float(f)
    g = float(g)
    h = float(h)
    i = float(i)

    # prediction
    result = model.predict(np.array([a, b, c, d, e, f, g, h,i]).reshape(1,-1))

    if result[0]==1:
        result = 'Fully Paid'
    else:
        result = 'Charged Off'
    prediction_text='Prediction: {}'.format(result)
    return render_template('index.html',result=prediction_text)
if __name__ == '__main__':
    app.run(debug=True)
