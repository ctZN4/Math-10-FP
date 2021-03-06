# Math 10 Final Project
# Author: Leo Cheung
# ID: 19421084

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.linear_model import LinearRegression

#change title of the page to FP-LC, and change icon to sunrise
#emoji code acquired from streamlit's github link at https://www.webfx.com/tools/emoji-cheat-sheet/

st.set_page_config(page_title = "Final Project - Leo Cheung", page_icon = ":sunrise:")
st.title("Math 10 - Analysis of CPI based on pricing of consumer goods")

st.markdown("Author: Leo Cheung, [GitHub link](https://github.com/ctZN4)")

df_show_hide = st.radio(
     "Do you wish to see the select snippets of data displayed throughout this page?",
     ('Show', 'Hide'))

df_toggle = (df_show_hide == "Show")

# P1
st.header("Data cleaning")

st.write("We import the CPI data acquired from the U.S. Bureau of Labor Statistics. ")

#Link to database: https://www.bls.gov/cpi/data.htm
#Downloaded from: https://download.bls.gov/pub/time.series/ap/
df = pd.read_table('ap.data.2.Gasoline.txt', sep='\s+')

#check and see that since the last column is all empty, it is dropped
if df["footnote_codes"].notna().sum() == 0:
    df = df.drop("footnote_codes", axis = 1)

#print df to display the data
if df_toggle:
    st.dataframe(df)
    see_df = "As we can see above, the dataframe"
else:
    see_df = "Our imported data"

st.write(see_df + " contains 3 main columns: the ID that describes the type of data, the year, month, and finally price of the product.")

st.write("The database also has two additional useful files: `ap.area` references the area code of the item, `ap.item` explains the types of products the item code.")

st.write("Thus, for the sake of clarity, we will splice the `series_id` into two additional columns for ease of access.")

df["area_code"] = df["series_id"].astype(str).str[3:7]
df["item_code"] = df["series_id"].astype(str).str[7:]

if df_toggle:
    st.write("Here's a snippet of the modified dataframe:")
    st.write(df.head(5))

df_avg = df[df["area_code"] == "0000"]

st.write("We will count the different types of gasoline this dataset contains:")

#gas_counts = df_avg["item_code"].value_counts()

st.bar_chart(df_avg["item_code"].value_counts())

#Copied from ap.item

#item_code  item_name
#74712	    Gasoline, leaded regular (cost per gallon/3.8 liters)
#74713	    Gasoline, leaded premium (cost per gallon/3.8 liters)
#74714	    Gasoline, unleaded regular, per gallon/3.785 liters
#74715	    Gasoline, unleaded midgrade, per gallon/3.785 liters
#74716	    Gasoline, unleaded premium, per gallon/3.785 liters
#74717	    Automotive diesel fuel, per gallon/3.785 liters
#7471A	    Gasoline, all types, per gallon/3.785 liters

st.write("Note that the type of gasoline with most data points is code `74714`, which refers to the price of unleaded regular gasoline per US gallon. We will isolate those values and sort them by year and month for further calculation.")

df_unl_reg = df_avg[df_avg["item_code"] == "74714"]
df2 = df_unl_reg.sort_values(by = ["year", "period"], axis = 0, ascending = True)


if df_toggle:
    st.write("Here is a preview of the modified dataset:")
    st.write(df2.head(5))

st.write("Checking the shape of the dataframe reveals that we have correctly chosen the type of unleaded gasoline, since the number of rows matches the number above for code `74714` (550).")
st.write(df2.shape)

st.write("We will now perform a series of operations that convert our `year` and `period` columns into one \"DT\" (i.e. datetime) column to represent the time.")
df2["DT"] = df2["year"].astype(str) + df2["period"]
df2["DT"] = df2["DT"].apply(lambda s: s[:4] + "-" + s[5:])
df2["DT"] = pd.to_datetime(df2["DT"])
if df_toggle:
    st.write("Checking the `DT` column of the dataframe shows that our conversion is successful.")
    st.write(df2["DT"].head(3).astype(str))



#P2
st.header("Regression")
st.write("In order to graph the relationship between of the gas price and time, we will use the days elapsed from the first day of year 1970 as a reference.")
df2["ref"] = pd.to_datetime("1970-01-01")
df2["diff"] = df2["DT"] - df2["ref"]
df2["diff"] = df2["diff"].apply(lambda t: int(str(t).split(" ")[0]))

if df_toggle:
    st.write("Below is a sample of the modified dataframe with gas price values and their corresponding date difference:")
    st.write(df2[["value", "diff"]].head(3))

chart_choice = st.selectbox(
    'Which regression graphs would you want to see?',
    ("Both regular and premium gasoline", "Regular gasoline only", "Premium gasoline only"))
view_chart = chart_choice[0]

reg = LinearRegression()
dfr = pd.DataFrame(df2["diff"].copy())
dfr = dfr.rename(columns={"diff":"x"})

#degree of polynomial
deg = 4
for i in range(deg):
    dfr[str("x"+str(i+1))] = dfr["x"] ** (i+1)

#remove column of "x" and keep the remaining "x_i"
dfr.drop(labels = "x", axis = 1)
def make_xi_list(d):
    return [f"x{i}" for i in range(1,d+1)]

reg.fit(dfr[make_xi_list(deg)], df2["value"])
theta0 = reg.intercept_
thetas = reg.coef_

brush = alt.selection_interval()


## Marking original data points
choiceX = "diff"
choiceY = "value"
gasChart = alt.Chart(df2).mark_circle().encode(
    x = alt.X(choiceX, scale=alt.Scale(zero=False)),
    y = alt.Y(choiceY, scale=alt.Scale(zero=False)),
    #NOTE: replaced simple colors with interactive chart, now selected points are colored and unselected are grey
    #color = alt.Color(choiceY, scale = alt.Scale(scheme = "greenblue")),
    color = alt.condition(brush, alt.Color("value:Q", scale = alt.Scale(scheme = "greenblue")), alt.value('grey')),
    opacity = alt.condition(brush, alt.value(1), alt.value(0.2)),
    tooltip = [choiceX, choiceY]
).properties(
    width = 800,
    height = 400
).add_selection(
    brush
)
gasChart.encoding.x.title = 'Time elapsed'
gasChart.encoding.y.title = 'Actual gas price'

## Graphing regression

x = df2["diff"].values.astype(float)

gas_reg = pd.DataFrame({
  'x': x,
  'y': theta0 + thetas[0] * x + thetas[1] * x ** 2 + thetas[2] * x **3 + thetas[3] * x **4
})

gas_reg_chart = alt.Chart(gas_reg).mark_line().encode(
    x = alt.X("x", scale=alt.Scale(zero=False)),
    y = alt.Y("y", scale=alt.Scale(zero=False)),
).properties(
    width = 800,
    height = 400
)
gas_reg_chart.encoding.x.title = 'Time elapsed'
gas_reg_chart.encoding.y.title = 'Predicted price'

if view_chart == "R" or view_chart == "B":
    st.write("The following Altair chart displays the data points of the gas prices and the regression line that represents the general trend.")
    st.altair_chart(gasChart + gas_reg_chart)

st.write("We want to find out the relationship between the regular and premium gasoline prices, and see if there is a correlation between the two. We will redo our previous operations on the unleaded regular gasoline dataset on the unleaded premium gasoline dataset.")

#choose premium, sort by time, then calculate time elapsed as "diff"
df3 = df_avg[df_avg["item_code"] == "74716"].copy()
df3["DT"] = df3["year"].astype(str) + df3["period"]
df3["DT"] = df3["DT"].apply(lambda s: s[:4] + "-" + s[5:])
df3["DT"] = pd.to_datetime(df3["DT"])
df3.sort_values(by = "DT", axis = 0, ascending = True)
df3["ref"] = pd.to_datetime("1970-01-01")
df3["diff"] = df3["DT"] - df3["ref"]
df3["diff"] = df3["diff"].apply(lambda t: int(str(t).split(" ")[0]))

#regression for premium gasoline

reg_p = LinearRegression()
dfr = pd.DataFrame(df3["diff"].copy())
dfr = dfr.rename(columns={"diff":"x"})

#degree of polynomial
deg = 4
for i in range(deg):
    dfr[str("x"+str(i+1))] = dfr["x"] ** (i+1)
    
dfr.drop(labels = "x", axis = 1)
def make_xi_list(d):
    return [f"x{i}" for i in range(1,d+1)]

reg_p.fit(dfr[make_xi_list(deg)], df3["value"])
theta0 = reg_p.intercept_
thetas = reg_p.coef_

## Marking premium gas data points
choiceX = "diff"
choiceY = "value"
gasChartP = alt.Chart(df3).mark_circle().encode(
    x = alt.X(choiceX, scale=alt.Scale(zero=False)),
    y = alt.Y(choiceY, scale=alt.Scale(zero=False)),
    #NOTE: replaced simple colors with interactive chart, now selected points are colored and unselected are grey
    #color = alt.Color(choiceY, scale = alt.Scale(scheme = "greenblue")),
    color = alt.condition(brush, alt.Color("value:Q", scale = alt.Scale(scheme = "turbo")), alt.value('grey')),
    opacity = alt.condition(brush, alt.value(1), alt.value(0.2)),
    tooltip = [choiceX, choiceY]
).properties(
    width = 800,
    height = 400
).add_selection(
    brush
)
gasChartP.encoding.x.title = 'Time elapsed'
gasChartP.encoding.y.title = 'Actual gas price'

## Graphing regression

x = df3["diff"].values.astype(float)

gas_reg = pd.DataFrame({
  'x': x,
  'y': theta0 + thetas[0] * x + thetas[1] * x ** 2 + thetas[2] * x **3 + thetas[3] * x **4
})

gas_p_chart = alt.Chart(gas_reg).mark_line().encode(
    x = alt.X("x", scale=alt.Scale(zero=False)),
    y = alt.Y("y", scale=alt.Scale(zero=False)),
).properties(
    width = 800,
    height = 400
)
gas_p_chart.encoding.x.title = 'Time elapsed'
gas_p_chart.encoding.y.title = 'Predicted price'

if view_chart == "P" or view_chart == "B":
    st.write("Below is the regression line for the premium gasoline data graphed on its data points.")
    st.altair_chart(gas_p_chart + gasChartP)

st.write("Note the similar shape of the two curves, it indicates that the two values are extremely closely related. So, we will attempt to calculate the difference between the two. We will first store the absolute difference in a new dataframe.")
df2s = df2[df2["diff"] >= df3["diff"].iloc[0]]
df_gas = pd.DataFrame(data = {"days": df3["diff"].values, "regular": df2s["value"].values, "premium": df3["value"].values})
df_gas["delta"] = df_gas["premium"] - df_gas["regular"]

if df_toggle:
    st.write(df_gas)

reg_diff = LinearRegression()
reg_diff.fit(np.array(df_gas["days"]).reshape(-1,1), df_gas["delta"])
theta0 = reg_diff.intercept_
thetas = reg_diff.coef_
st.write("Calculating a regression for the difference of gas prices, we get an intercept of " + str(theta0) + " and coefficient of " + str(thetas[0]))
st.write("Therefore, we can conclude that the difference between the two types of gas prices as time goes on is almost constant, and the increased difference can be accounted for with inflation.")


#P3
st.header("References")

st.write("The project is completed based on the materials covered in the course, and I did not acquire help from external sources other than the instructions found in the documentation of each library, including Pandas, Altair, Sci-kit Learn and Streamlit.")
st.markdown("As stated above, the [CPI database](https://www.bls.gov/cpi/data.htm) is acquired from the US Bureau of Labor Statistics.")