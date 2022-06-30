"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
# Data dependencies
from sklearn.pipeline import Pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import streamlit.components.v1 as components
import pandas as pd
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import os



# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

st.set_option('deprecation.showPyplotGlobalUse', False)


# Load your raw data
raw = pd.read_csv("resources/train.csv")
train_df = raw



# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	html_temp = """
	<div style="background-color:{};padding:10px;border-radius:10px;margin:10px;">
	<h1 style="color:{};text-align:center;">EDSA - Climate Change Belief Analysis 2022</h1>
	</div>
	"""

	title_temp ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h1 style="color:white;text-align:center;">Classification Predict</h1>
	<h2 style="color:white;text-align:center;">JS4</h2>
	</div>
	"""
	

	# Creates a main title and subheader on your page -
	# these are static across all pages


	# Creating sidebar with selection box -
	# you can create multiple pages this way
	
	menu = ["Home", "About The Predict", "Exploratory Data Analysis", "Modelling","Model Performance"]
	selection = st.sidebar.selectbox("Menu", menu)

	if selection == "Home":
		st.markdown(html_temp.format('black','white'), unsafe_allow_html=True)
		st.markdown(title_temp, unsafe_allow_html=True)
		
   
                        

	# Building out the "About" page
	if selection == "About The Predict":
		markup(selection)
		st.info("Project Oveview : Using a individuals historical tweet data predict that individuals belief on climate change")
		
		# You can read a markdown file from supporting resources folder
		if st.checkbox("Introduction"):
			st.subheader("Introduction to Classification Predict")
			st.info("""With the rise of climate change many companies have the goal of lessening their their own carbon footprint and their enviromental impact.In this predict aim to to determine peoples sentiments, whether they believe climate change is a real threat or not. We will use machine learning models to determine the above mentioned.
			 They would like to determine how people perceive climate change and whether or not they believe it is a real threat or not.We will use machine learning models to determine the above mentioned (techniques used are Logistic Regression, Support Vector Classification and LinearSVC)
			 """)

		
		if st.checkbox("Problem Statement"):
			st.subheader("Problem Statement of the Classification Predict")
			st.info("The aim is to build natural language processing models that determine whether that particular individual beliefs in climate change")

		if st.checkbox("Data"):
			st.subheader("Data Decription and Data Source")
			st.info("""Data will be sourced from the Canada Foundation for Innovation JELF grant to Chris Bauch University of Waterloo. 43943 tweets were collected. Tweet will be labelled into one of the following classes:
			 
>2- News: the tweet links to factual news about the climate change

>1 - Pro: the tweet support the believe of man made climate change

>0 - Neutral: the tweet neither supports nor refutes the believe of manmade climate change

>-1 - Anti: the tweet does not believe in man made climate change

Variable definitions:
>sentiment: (Sentiment of tweet)
>message: (Tweet body)
>tweetid: (Twitter unique id)""")

			st.subheader("Raw Twitter data and label")
			if st.checkbox('Show raw data'): # data is hidden if box is unchecked
				st.write(raw[['sentiment', 'message']]) # will write the df to the page



	# Building out the predication page
	if selection == "Modelling":
		markup(selection)
		option = st.selectbox('Select a model',['SVC','LinearSVC','Logistic Regression'])
		if option == 'LinearSVC':
			st.info('Linear SVC - it fits to the data you provide, returning a "best fit" hyperplane that divides, or categorizes, your data. From there, after getting the hyperplane, you can then feed some features to your classifier to see what the "predicted" class is.')
					# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")
				
		if option == 'Logistic Regression':
			st.info("Logistic Regression - A machine learning method that computes the probability of an event occuring and places it in the relevant class or category")
					# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")
				

			if st.button("Classify"):
						# Transforming user input with vectorizer
						#vect_text = tweet_cv.fit_transform([tweet_text]).toarray()
						# Load your .pkl file with the model of your choice + make predictions
						# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])

						# When model has successfully run, will print prediction
						# You can use a dictionary or similar structure to make this output
						# more human interpretable.
				st.success("Text Categorized as: {}".format(prediction))

				if prediction[0] == 2:
					st.success('Tweet links factual news about climate change')
				if prediction[0] == 1:
					st.success('Tweet support the believe of man-made climate change')
				if prediction[0] == 0:
					st.success('Tweet neither supports nor refutes the believe of man-made climate change')
				if prediction[0] == -1:
					st.success('Tweet does not believe in man-made climate change')
			if st.button("Classify"):
						# Transforming user input with vectorizer
						#vect_text = tweet_cv.fit_transform([tweet_text]).toarray()
						# Load your .pkl file with the model of your choice + make predictions
						# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/LinearSVC_model.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])

						# When model has successfully run, will print prediction
						# You can use a dictionary or similar structure to make this output
						# more human interpretable.
				st.success("Text Categorized as: {}".format(prediction))

				if prediction[0] == 2:
					st.success('Tweet links factual news about climate change')
				if prediction[0] == 1:
					st.success('Tweet support the believe of man-made climate change')
				if prediction[0] == 0:
					st.success('Tweet neither supports nor refutes the believe of man-made climate change')
				if prediction[0] == -1:
					st.success('Tweet does not believe in man-made climate change')

		

		

		
				

			if st.button("Classify"):
						# Transforming user input with vectorizer
						#vect_text = tweet_cv.fit_transform([tweet_text]).toarray()
						# Load your .pkl file with the model of your choice + make predictions
						# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/KNeighborsClassifier.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])

						# When model has successfully run, will print prediction
						# You can use a dictionary or similar structure to make this output
						# more human interpretable.
				st.success("Text Categorized as: {}".format(prediction))

				if prediction[0] == 2:
					st.success('Tweet links factual news about climate change')
				if prediction[0] == 1:
					st.success('Tweet support the believe of man-made climate change')
				if prediction[0] == 0:
					st.success('Tweet neither supports nor refutes the believe of man-made climate change')
				if prediction[0] == -1:
					st.success('Tweet does not believe in man-made climate change')

		
				

			

		

		if option == 'Logistic Regression':
			st.info("Logistic Regression - A machine learning method that computes the probability of an event occuring and places it in the relevant class or category")
					# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")
				

			if st.button("Classify"):
						# Transforming user input with vectorizer
						#vect_text = tweet_cv.fit_transform([tweet_text]).toarray()
						# Load your .pkl file with the model of your choice + make predictions
						# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])

						# When model has successfully run, will print prediction
						# You can use a dictionary or similar structure to make this output
						# more human interpretable.
				st.success("Text Categorized as: {}".format(prediction))

				if prediction[0] == 2:
					st.success('Tweet links factual news about climate change')
				if prediction[0] == 1:
					st.success('Tweet support the believe of man-made climate change')
				if prediction[0] == 0:
					st.success('Tweet neither supports nor refutes the believe of man-made climate change')
				if prediction[0] == -1:
					st.success('Tweet does not believe in man-made climate change')

	







	# Building out the EDA page
	if selection == "Exploratory Data Analysis":
		markup(selection)
		train_df = raw


		#plot and visualize the most used words in wordcloud
		if st.checkbox("Summary"):
			st.subheader("Summary of the data")
			st.info("The data is in distributed in four groups :1 being News with 8530 tweets, 2 being Pro or positive sentiment with 3640 tweets, 0 being Neutral sentiment with 2533 tweets and lastly -1 anti sentiment with 1296 tweets. We can see that we are dealing with imbalanced which may be a problem when we are training our model.We also see that the positive sentiment has the highest number of entries.Which we might think that most people in our data believe in climate change ,We need to do more digging to come to a conclusion .")

			


		



		if st.checkbox("Metrics"):
			st.subheader('Pearson Correlation Coeficient Figure')
			corr = train_df.corr(method = 'pearson')
			f, ax = plt.subplots(figsize=(11, 9))
			cmap = sns.diverging_palette(10, 275, as_cmap=True)
			sns.heatmap(corr, cmap=cmap, square=True,
			linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=ax)
			st.pyplot()
			st.info("From this we can't find any correlation between message length and sentiment")


	# Building out the "Model Perfomance" page
	if selection == "Model Performance":
		markup(selection)
		st.subheader('Summary of all our models perfomance')
		st.info("Model performance will be judged using two metrics the f1 score we have that logistic regresion after hyperparameter tuning gives a score of 0.78 and linearSVC gives 0.74")
		
		


		


def markup(heading):
	html_temp = """<div style=background-color:{};padding:10px;boarder-radius:10px"><h1 style="color:{};text-align:center;">"""+heading+"""</h1>"""
	st.markdown(html_temp.format('royalblue','white'), unsafe_allow_html=True)


#Function to lable our Sentiments
def getAnalysis(score):
    if score == 2:
        return 'News'
    elif score == 1:
        return 'Pro'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Anti'
train_df['Analysis'] = train_df['sentiment'].apply(getAnalysis)
train_df['msg_len'] = train_df['message'].apply(lambda x: len(x))


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
	main()