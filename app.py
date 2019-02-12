from flask import Flask, request, render_template, jsonify, Response
import requests
import pandas as pd
import pickle
import os
import numpy as np
import datetime


app = Flask(__name__)



dirname = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(dirname,'data/BOS_app_data.csv'), index_col='id')
df = df.drop(labels = ['Unnamed: 0', 'Unnamed: 0.1', 'name', 'zipcode', 'lat', 'lng', 'rating_n', 'prim_type'], axis = 1)
df.fillna(value=0, inplace=True)

name_df =  pd.read_csv(os.path.join(dirname,'data/max_times_forBOSTON.csv'), index_col='id')
day = datetime.datetime.today().isoweekday()
if day == 7:
	df_day = 0
else:
	df_day = day

model = pickle.load(open("model_rf.pkl","rb"))
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
				response = 'Moderate (Noisy, but manageable for conversation)'
				name = name_df.loc[str(check_id), 'name']	
				x = name_df.loc[str(check_id), str(df_day)]
			if prediction == 1:
				response = 'Loud (Potentially distracting/uncomfortable)'
				name = name_df.loc[str(check_id), 'name']
				x = name_df.loc[str(check_id), str(df_day)]
			if prediction == 2:
				response = 'Very Loud (Lilkely distracting/uncomfortable)'
				name = name_df.loc[str(check_id), 'name']
				x = name_df.loc[str(check_id), str(df_day)]
		except:
			response = 'No prediction available for this location'
			x = ' '
			name = request.form['placename']
		return render_template('prediction.html.j2', responsetext=response, busytime = x, name=name) 




if __name__ == "__main__":
	app.run(debug=True)
	# port = int(os.environ.get('PORT', 5000))
	# app.run(host='0.0.0.0', port=port)
