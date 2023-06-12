from flask import Flask, render_template, url_for, request, flash, redirect
from models import get_auth, Trait, plot, t, prob
import numpy as np
from flask_login import login_user, logout_user, login_required
from models import db, User, login_manager
import tweepy

app = Flask(__name__)
app.config['SECRET_KEY'] = '$uper$secreT'
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///./users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
auth = get_auth()
trait = Trait()

db.init_app(app)
login_manager.init_app(app)

with app.app_context():
    db.create_all()


@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == 'POST':
        a = request.form['value']
        if a:
            try:
                auth.auth1.get_access_token(a)
                me = auth.api.verify_credentials()
                auth.screen_name = me.screen_name
                query = User.query.filter_by(username=auth.screen_name)
                try:
                    user = query.one()
                except:
                    user = User(username=auth.screen_name)
                    db.session.add(user)
                    db.session.commit()
                login_user(user)
                return redirect(url_for("search"))
            except:
                auth.regenerate()
                return render_template("home.html", link=auth.redirect_url)
        else:
            auth.regenerate()
            return render_template("home.html", link=auth.redirect_url)
    auth.regenerate()
    return render_template("home.html", tlink=auth.redirect_url)


@app.route('/search', methods=["GET", "POST"])
@login_required
def search():
    if request.method == 'POST':
        if request.form['submit_button'] == 'submit':
            username = request.form['User_name']
            try:
                user = auth.api.get_user(screen_name=username)
                return redirect(url_for('result', username=username))
            except:
                return redirect(url_for("search"))
        elif request.form['submit_button'] == 'submit_a':
            username = auth.screen_name
            return redirect(url_for('result', username=username))
    return render_template("search.html")


@app.route('/result', methods=["GET"])
@login_required
def result():
    username = request.args.get('username')
    usertrait = trait.predict(username, auth)
    plot(usertrait, prob[np.array(usertrait).argmax()])
    link = "https://twitter.com/" + username
    return render_template("result.html", username=username, trait=t[np.array(usertrait).argmax()][0],
                           desc=t[np.array(usertrait).argmax()][1], link=link)


@app.route("/logout", methods=["GET", "POST"])
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))


if __name__ == '__main__':
    app.run(debug=True)
