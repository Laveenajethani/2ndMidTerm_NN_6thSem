import streamlit as st
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)

# Load the pickled model
model = pickle.load(open('NN_model.pkl', 'rb')) 
dataset= pd.read_csv('PCA and NN Dataset2.csv')

# Extracting dependent and independent variables:
# Extracting independent variable:
x = dataset.iloc[:,0:9].values
# Extracting dependent variable:
y = dataset.iloc[:, -1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)

#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(x[:, : ]) 

#Replacing missing data with the calculated mean value  
x[:, : ]= imputer.transform(x[:, : ]) 

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
y = labelencoder_x.fit_transform(y)
y = labelencoder_x.fit_transform(y)



def GenderVoicePredictor(meanfreq,sd,median,IQR,skew,kurt,mode,centroid,dfrange): 
  output= model.predict(sc.transform([[meanfreq,sd,median,IQR,skew,kurt,mode,centroid,dfrange]]))
  print(output)
  return output

def main():

    html_temp = """
   <div class="" style="background-color:gray;" >
   <div class="clearfix">
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">2nd Mid Term</p></center>
   <center><p style="font-size:30px;color:white;margin-top:10px;">Laveena Jethani</p></center>
   <center><p style="font-size:25px;color:white;margin-top:0px;">Sec:B , PIET18CS080 </p></center>
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Voice Prediction")
    meanfreq = st.number_input('Insert meanfreq')
    sd = st.number_input('Insert sd')
    median = st.number_input('Insert median')
    IQR = st.number_input('Enter IQR')
    skew = st.number_input('Enter skew')
    kurt = st.number_input('Enter kurt')
    mode = st.number_input('Enter mode')
    centroid = st.number_input('Enter centroid')
    dfrange = st.number_input('Enter dfrange')


    result=""
    if st.button("Predict"):
      result=GenderVoicePredictor(meanfreq,sd,median,IQR,skew,kurt,mode,centroid,dfrange)
      st.success('Model has predicted that -> {}'.format(result))
    if st.button("About"):
      st.subheader("Laveena Jethani")
      st.subheader("Computer Science,PIET")

if __name__=='__main__':
  main()
