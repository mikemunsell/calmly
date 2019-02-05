from flask import Flask, request, render_template, jsonify, Response
import requests
import pandas as pd
import pickle
import os
import numpy as np


app = Flask(__name__)



dirname = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(dirname,'data/fulldf_example.csv'), index_col='place_id')
df = df.drop(labels = ['Unnamed: 0', 'name', 'zipcode', 'lat', 'lng', 'prim_type', 'noise_level', 'lograting_n', 'rating_n_x_pop'], axis = 1)
df.fillna(value=0, inplace=True)
model = pickle.load(open("model.pkl","rb"))
API_KEY = 'AIzaSyAmzxG3BdFSxoQh61MypS_s2puE7Ud9MyU'

@app.route('/', methods = ['GET', 'POST'])
def index():
	key=API_KEY
	return render_template('savedhtml.html.j2',key=key)

@app.route('/predict', methods = ['GET', 'POST'])
def results():
	if request.method == 'POST':
		try:
			check_id = request.form['placeid']
			predict_df = df.loc[str(check_id),:]
			prediction = model.predict(predict_df.values.reshape(1, -1))[0]
			if prediction == 0:
				response = 'Moderate'
				x = 30	
			if prediction == 1:
				reponse = 'Loud'
				x = 60
			if prediction == 2:
				response = 'Very Loud'
				x = 90
		except:
			response = 'Not enough data for prediction for this location'
			x = 0
	payload = {"data": x, "noise:" reponse}
	return Response(jsonify(payload), mimetype= 'text/event-stream')



# @app.route('/predict', methods = ['GET', 'POST'])
# def results():
# 	if request.method == 'POST':
# 		check_id = request.form['placeid']
# 		predict_df = df.loc[str(check_id),:]
# 		if len(predict_df) == 0:
# 			response = 'Sorry, there is not enough data to predict the noise level at this location'
# 		else:
# 			prediction = model.predict(predict_df.values.reshape(1, -1))[0]
# 			if prediction == 0:
# 				response = 'Moderate'
# 			if prediction == 1:
# 				reponse = 'Loud'
# 			if prediction == 2:
# 				response = 'Very Loud'
# 		return render_template('prediction.html.j2', responsetext=response)
	


if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)
