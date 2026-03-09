import numpy as np
import streamlit as st
import pickle

# load the saved trained model
with open("models/trained_model.sav", "rb") as f:
    loaded_trained_model = pickle.load(f)

# loading the saved scaler
with open("models/labelencoder.sav", "rb") as f:
    loaded_trained_model = pickle.load(f)

# loading the saved scaler
with open("models/standardscaler.sav", "rb") as f:
    loaded_scaler = pickle.load(f)

# loading the saved scaler
with open("models/one_hot_encoder.sav", "rb") as f:
    loaded_one_hot_encoder = pickle.load(f)