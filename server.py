from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)

# Import model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))


# Prediction API route
@app.route("/predict", methods=['GET'])
def prediction():
    args = request.args
    company = args.get('company')
    laptopType = args.get('type')
    ram = args.get('ram')
    weight = args.get('weight')
    touchScreen = args.get('touchScreen')
    ips = args.get('ips')
    screenSize = args.get('screenSize')
    resolution = args.get('screenResolution')
    cpu = args.get('cpu')
    hdd = args.get('hdd')
    gpu = args.get('gpu')
    os = args.get('os')
    ssd = args.get('ssd')

    ppi = None

    if touchScreen == 'yes':
        touchScreen = 1
    else:
        touchScreen = 0
    
    if ips == 'yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])

    ppi = float(((X_res**2) + (Y_res**2))**0.5)/float(screenSize)

    query = np.array([company, laptopType, ram, weight, touchScreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 12)

    predicted_value = int(np.exp(pipe.predict(query)[0]))

    return str(predicted_value)

if __name__ == "__main__":
    app.run(debug=True)