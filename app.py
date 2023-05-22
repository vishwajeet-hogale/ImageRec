from flask import Flask, request, make_response
from services import services

classifier = None
ResultMap = None


app = Flask(__name__)

app.route("/predict",methods = ["POST"])
def predict():
    if not classifier or not ResultMap :
        classifier,ResultMap = services.load_model()
    if 'image' not in request.files:
        return 'No image file in the request', 400
    image_file = request.files['image']
    save_path = './Output/img.jpg'
    image_file.save(save_path)
    prediction = services.predict_image(classifier,ResultMap,save_path)
    response = make_response(prediction)
    response.headers['Content-Type'] = 'text/plain'
    return response

if __name__ == "__main__":
    app.run()