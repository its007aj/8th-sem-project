from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/details", methods=['GET', 'POST'])
def details():
    name = [str(x) for x in request.form.values()]
    return render_template('form.html', name=name[0])

@app.route("/destination", methods=['GET','POST'])
def destination():
    l = [str(x) for x in request.form.values()]
    #print(l)
    #l[0] city
    #l[1] hotelname
    #l[2] description
    import model
    d = model.predict(l[0], l[1], l[2])
    if d.empty:
        return render_template('home_again.html')
    else:
        return render_template('display.html',  tables=[d.to_html(classes='data')], titles=['You can check these out'])
if __name__=="__main__":
    app.run()
