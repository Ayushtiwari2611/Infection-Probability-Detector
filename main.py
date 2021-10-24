from logging import debug
from os import name
from flask import Flask, render_template, request
app = Flask(__name__)
import pickle

# Open a file, where you want to store the data
file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        bodyPain = int(myDict['pain'])
        runnyNose = int(myDict['runnyNose'])
        difficultyBreathing = int(myDict['difficultyBreathing'])
        # Code with inference
        infectionProbablity = clf.predict_proba([[fever, bodyPain, age, runnyNose, difficultyBreathing]])[0][1]
        print(infectionProbablity)
        return render_template('show.html', inf=round(infectionProbablity*100))
    return render_template('index.html')
    # return 'Hello, World!' + str(infectionProbablity)

if __name__ == "__main__":
    app.run(debug=True)