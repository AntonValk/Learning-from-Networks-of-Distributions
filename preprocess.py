import numpy as np
import pandas as pd



def preprocess_data_geog():
	df = pd.read_json(open("train.json", "r"))

	# df = df[df.latitude >= 40]
	# df = df[df.latitude <= 42]
	# df = df[df.longitude >= -75]
	# df = df[df.longitude <= -73]

	df["num_photos"] = df["photos"].apply(len)
	df["num_features"] = df["features"].apply(len)
	df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
	df["created"] = pd.to_datetime(df["created"])
	df["created_month"] = df["created"].dt.month
	df["created_day"] = df["created"].dt.day

	return df