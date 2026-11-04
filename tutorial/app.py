# @file app.py
# @desc Driver for navigating tutorial pages as web app 


from flask import Flask, render_template, redirect, request # server + navigation 
import json 
import sys
import os 
import time

# Create an app from this driver file
# __name__ = main when this script is called directly

app = Flask(__name__)

# Homepage
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/manual")
def manual():
    return render_template("manual.html")

@app.route("/experiment1")
def exp1():
    return render_template("experiment1.html")

@app.route("/experiment2")
def exp2():
    return render_template("experiment2.html")

@app.route("/experiment3")
def exp3():
    return render_template("experiment3.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

