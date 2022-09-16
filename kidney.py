import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

# Loaded the model
model=pickle.load(open('kidney final.pkl','rb'))



@app.route('/')
def home():
    return render_template('kidney.html')


# @app.route('/heart_api',methods=['POST'])
# def heart_api():
#     data=request.json['data']   # The input is given in json format and it will be captured and stored in data variable 
#     # The data will be in key-value pairs 
#     print(data)
#     arrdata=np.array(list(data.values()))
#     arrdata.astype(float)
#     #intdata=int(arrdata)
#    # print(np.array(list(data.values())).reshape(1,-1))
#     data1=arrdata.reshape(1,-1)
#     output=model.predict(data1)
#     print(int(output[0]))
#     return jsonify(int(output[0]))
    


@app.route('/predict3',methods=['POST'])
def predict3():
    data1=[float(x) for x in request.form.values()]
    data1=np.array(data1).reshape(1,-1)
    output=model.predict(data1)[0]
    if output==1:
         return render_template("kidney.html",prediction_text="The person is having kidney disease")
    else:
        return render_template("kidney.html",prediction_text="The person is safe")





if __name__=="__main__":
        app.run(debug=True)




