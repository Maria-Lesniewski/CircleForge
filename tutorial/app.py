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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

