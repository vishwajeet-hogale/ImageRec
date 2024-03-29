import os
from flask import Flask, request, make_response,render_template,request,flash,url_for,redirect
from Model import services

FaceDict = {}
keys = {}

app = Flask(__name__)

# app.config['SECRET_KEY'] = 'the random string'

@app.route("/",methods=["GET"])
def index():
    return render_template("index.html",keys = keys)
@app.route("/predict",methods = ["POST","GET"])
def predict():
    global FaceDict
    # if not FaceDict :
    #     FaceDict = services.train()
    if request.method == "POST":
        name = request.form['name']
        # return {"Hello":all_image_paths = FaceDict[request.form["name"]]}
        pred_list = None
        try:
            pred_list = FaceDict[str(name)]
        except:
            FaceDict = services.load_face_dict("./Output")
            pred_list = FaceDict[name]

            # services.copy_files(name,[i.split("/")[-1] for i in pred_list])
        pred_list = [i.split("/")[-1] for i in pred_list]
        services.copy_files(name,pred_list)
        return render_template("index.html",all_image_paths = pred_list,keys=FaceDict.keys())
    return render_template("index.html",keys = keys)

@app.route("/predict1",methods = ["POST","GET"])
def predict1():
    global FaceDict
    # if not FaceDict :
    #     FaceDict = services.train()
    if request.method == "POST":
        name = request.form['name']
        # return {"Hello":all_image_paths = FaceDict[request.form["name"]]}
        pred_list = None
        try:
            pred_list = FaceDict[str(name)]
        except:
            FaceDict = services.load_face_dict("./Output")
            pred_list = FaceDict[name]

            # services.copy_files(name,[i.split("/")[-1] for i in pred_list])
        pred_list = [i.split("/")[-1] for i in pred_list]
        services.copy_files(name,pred_list)
        return render_template("index.html",all_image_paths = pred_list,keys=FaceDict.keys())
    return render_template("index.html",keys = keys)


if __name__ == "__main__":
    # global FaceDict
    # global keys
    if not FaceDict:
        if not os.path.exists("./Output/FaceDict.pkl"):
            FaceDict = services.train()
        else:
            FaceDict = services.load_face_dict("./Output")
    services.write_face_dict(FaceDict,"./Output")
    keys = FaceDict.keys()
    app.run(debug = False,port = 5000)