# Math 10 Final Project
# Author: Leo Cheung
# ID: 19421084

import numpy as np
import pandas as pd
import altair as alt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import streamlit as st

st.set_page_config(page_title = "Final Project - Leo Cheung", page_icon = ":sunrise:")
st.title("Test")

st.markdown("Author: Leo Cheung, [GitHub link](https://github.com/ctZN4)")

df = pd.read_table('ap.data.2.Gasoline.txt', sep='\s+')

st.write(df)

ifff = df.info()

st.write(type(df.info()))

st.write(ifff)