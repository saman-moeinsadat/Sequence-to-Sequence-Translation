from flask import Flask, redirect, url_for, request
from translate_detect import translate

app = Flask(__name__)


@app.route('/',methods = ['POST', 'GET'])
def login():

    if request.method == 'GET':

        return """
                <h1>French to English Translator</h1><p>This site is a prototype API for
                French to English Machine Translation. Pleasee note that the sentences 
                must not exceed 8 words.</p>
                <form method="POST">
                <h3>Please Enter The Sentence:</h3>
                <input name="sentence"><input type="submit"></form>    
            """
    elif request.method == 'POST':

        sentence = request.form["sentence"]
        translated_sentence = translate(sentence)

        return """<h2>Translated Sentence: </h2>
                <h3>{}</h3>
            """.format(translated_sentence)

    




if __name__ == "__main__":
    app.run(debug = True)
