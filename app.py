#!/usr/bin/env python
# coding: utf-8


from __future__ import absolute_import, division, print_function, unicode_literals


import pickle
from pickle import load
from pickle import dump

from tensorflow import keras
import numpy as np
from flask import Flask, request, jsonify, render_template,redirect,flash,send_file

import os
import urllib.request


from flask import Flask , render_template , request , redirect, url_for,flash
from werkzeug.utils import secure_filename
from sudoku_main import sudoku_crop_solve_save


import cv2


solution, existing_numbers ,sudoku , cropped_sudoku = 0,0,0,0
raw_img_count=0
raw_img_path=r"static\img\sudoku\raw_sudoku_{count}.jpg".format(count=str(raw_img_count))
crop_img_path=r"static\img\sudoku\cropped_sudoku_{count}.jpg".format(count=str(raw_img_count))
img_count = 0
img_name = "solved_cropped{no}_sudoku_{count}.jpg".format(no=raw_img_count, count=img_count)
img_path = r"static\img\sudoku\{}".format(img_name)
active_num=""
def sudoku_ready():
    global solution , existing_numbers , sudoku ,cropped_sudoku ,raw_img_count , img_count , active_num
    solution , existing_numbers , sudoku ,cropped_sudoku = sudoku_crop_solve_save(raw_img_count , required_num_in_sol="0")
    print(sudoku)
    if sudoku :
        img_count=0
        active_num=""
        print("sudoku_ready")
        return True
    else :
        return False

def sudoku_filter_sol(req_num):
    if (req_num=="All"):
        req_num="123456789"
    global img_count , img_path , img_name
    img_count=req_num
    img_name = "solved_cropped{no}_sudoku_{count}.jpg".format(no=raw_img_count, count=img_count)
    img_path = r"static\img\sudoku\{}".format(img_name)

    cropped_sudoku_copy = cropped_sudoku.copy() 
    sudoku.write_solution(cropped_sudoku_copy, solution, ignore=existing_numbers ,required_num_in_sol =req_num)
    cropped_sudoku_copy = cv2.resize(cropped_sudoku_copy , (450,450))
    cv2.imwrite( img_path, cropped_sudoku_copy )
    return True


#from neural_model import NeuralModel
#NeuralModel.instance()
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
@app.route("/")
@app.route("/home")
def home():
    mydir = r"static/img/sudoku"
    if (len(os.listdir(mydir))>0):
        for f in os.listdir(mydir):
            os.remove(os.path.join(mydir, f))
    return render_template('index.html' , page="home")

@app.route("/confirm" , methods=["GET" , "POST"])
def confirm():
    global raw_img_count
    print("Inside (confirm) raw_img_count = ",raw_img_count)
    raw_img_path=r"static\img\sudoku\raw_sudoku_{count}.jpg".format(count=str(raw_img_count))
    crop_img_path=r"static\img\sudoku\cropped_sudoku_{count}.jpg".format(count=str(raw_img_count))
    if request.method=="POST":
        img = request.files['sudoku_raw_img']
        if(img.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS) :
            img.save(raw_img_path)
            if(sudoku_ready()):
                print("sudoku ready")
                return render_template('confirm.html',page="confirm", crop_img_path=crop_img_path)
                
            else:
                print("Sudoku Not Ready")
                #flash("Sudoku Not Ready")
                return redirect(url_for("upload"))
        else:
            print("wrong file format")
            #flash("wrong file format")
            return redirect(url_for("upload"))
    else:
        print("not a post request")
#--------------------------------------------------------

@app.route("/confirm_again")
def confirm_again():
    global raw_img_count , crop_img_path
    print("Inside (confirm_again) raw_img_count = ",raw_img_count)
    raw_img_count=raw_img_count + 1
    
    raw_img_path=r"static\img\sudoku\raw_sudoku_{count}.jpg".format(count=str(raw_img_count))
    cv2.imwrite(raw_img_path , cropped_sudoku)
    crop_img_path=r"static\img\sudoku\cropped_sudoku_{count}.jpg".format(count=str(raw_img_count))
    if(sudoku_ready()):
        print("sudoku ready")
        return render_template('confirm_again.html',page="confirm", crop_img_path=crop_img_path)
        
    else:
        print("Sudoku Not Ready")
        #flash("Sudoku Not Ready")
        return redirect(url_for("upload"))

#---------------------------------------------------------

@app.route("/upload")
def upload():
    global raw_img_count 
    raw_img_count = raw_img_count + 1
    print("Inside (upload) raw_img_count = ",raw_img_count)
    return render_template('upload.html',page="upload")


@app.route("/result" )
def result():
    global raw_img_count , img_path , img_name
    print("Inside (upload) raw_img_count = ",raw_img_count)

    img_name = "solved_cropped{no}_sudoku_{count}.jpg".format(no=raw_img_count, count=img_count)
    img_path = r"static\img\sudoku\{}".format(img_name)

    return render_template('result.html' , img_path=img_path , active_num=active_num ,page="result")


@app.route("/result/<string:num>" )
def result_filter(num="All"):
    global raw_img_count
    print("Inside (result_filter) raw_img_count = ",raw_img_count)
    if solution==0:
        sudoku_ready()
    print("in result_filter route")
    global active_num
    if(num):
        active_num = num
        print("required_num_in_sol = "+num)
        if(sudoku_filter_sol(num)):
            print("solution filtered successfully")
    else:
        active_num=""
        sudoku_filter_sol("All")

    return redirect(url_for('result'))
    


@app.route("/guide")
def guide():
    return render_template('instructions.html' , page="guide")

if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.run(debug=False )
