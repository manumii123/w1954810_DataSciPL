import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Global_Superstore", page_icon="Data Analysis:", layout="wide")

st.title("Data Analysis:Global_Superstore")

df = pd.read_csv("GlobalSuperstore.csv", encoding="ISO-8859-1")

col1, col2 = st.columns((2))
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Getting the min and max date
startDate = pd.to_datetime(df["Order Date"]).min()
endDate = pd.to_datetime(df["Order Date"]).max()

with col1:
    date1 = pd.to_datetime(st.date_input("Start Date", startDate))

with col2:
    date2 = pd.to_datetime(st.date_input("End Date", endDate))

df = df[(df["Order Date"] >= date1) & (df["Order Date"] <= date2)].copy()

st.sidebar.header("Pick Your Filter: ")

# Filters for Category and Sub-Category
category = st.sidebar.multiselect("Pick your Category", df["Category"].unique())
if not category:
    df2 = df.copy()
else:
    df2 = df[df["Category"].isin(category)]

subcategory = st.sidebar.multiselect("Pick your Sub-Category", df2["Sub-Category"].unique())
if not subcategory:
    df3 = df2.copy()
else:
    df3 = df2[df2["Sub-Category"].isin(subcategory)]

filtered_df = df3.copy()

# Create for Region
region = st.sidebar.multiselect("Pick your Region", df["Region"].unique())
if not region:
    df2 = df.copy()
else:
    df2 = df[df["Region"].isin(region)]

# Create for State
state = st.sidebar.multiselect("Pick the State", df2["State"].unique())
if not state:
    df3 = df2.copy()
else:
    df3 = df2[df2["State"].isin(state)]

# Create for City
city = st.sidebar.multiselect("Pick the City", df3["City"].unique())

# Filter the data based on Region, State and City

if not region and not state and not city:
    filtered_df = df
elif not state and not city:
    filtered_df = df[df["Region"].isin(region)]
elif not region and not city:
    filtered_df = df[df["State"].isin(state)]
elif state and city:
    filtered_df = df3[df["State"].isin(state) & df3["City"].isin(city)]
elif region and city:
    filtered_df = df3[df["Region"].isin(region) & df3["City"].isin(city)]
elif region and state:
    filtered_df = df3[df["Region"].isin(region) & df3["State"].isin(state)]
elif city:
    filtered_df = df3[df3["City"].isin(city)]
else:
    filtered_df = df3[df3["Region"].isin(region) & df3["State"].isin(state) & df3["City"].isin(city)]

category_df = filtered_df.groupby(by=["Category"], as_index=False)["Sales"].sum()

# create pie charts for region sales and category sales
with st.container():
    with col1:
        st.subheader("Region-Sales")
        fig2 = px.pie(filtered_df, values="Sales", names="Region", hole=0.5)
        fig2.update_traces(text=filtered_df["Region"], textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Category-Sales")
        fig1 = px.pie(category_df, values="Sales", names="Category")
        fig1.update_traces(text=category_df["Category"], textposition="outside")
        st.plotly_chart(fig1, use_container_width=True)


    cl1, cl2 = st.columns((2))
with cl1:
    with st.expander("Region_ViewData"):
        region = filtered_df.groupby(by="Region", as_index=False)["Sales"].sum()
        st.write(region.style.background_gradient(cmap="Oranges"))
        csv = region.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", data=csv, file_name="Region.csv", mime="text/csv",
                           help='Click here to download the data as a CSV file')
with cl2:
    with st.expander("Category_ViewData"):
        st.write(category_df.style.background_gradient(cmap="Blues"))
        csv = category_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", data=csv, file_name="Category.csv", mime="text/csv",
                           help='Click here to download the data as a CSV file')



# Association Rules 
data = {
    "antecedents": ["Accessories", "Tables", "Appliances", "Accessories", "Supplies", "Bookcases", "Phones", "Chairs",
                    "Machines", "Accessories", "Phones", "Storage", "Copiers"],
    "consequents": ["Tables", "Accessories", "Accessories", "Appliances", "Bookcases", "Supplies", "Chairs", "Phones",
                    "Accessories", "Machines", "Appliances", "Copiers", "Storage"],
    "antecedent_support": [0.08, 0.06, 0.12, 0.04, 0.02, 0.16, 0.16, 0.04, 0.08, 0.04, 0.24, 0.06, 0.20],
    "consequent_support": [0.06, 0.08, 0.04, 0.12, 0.16, 0.02, 0.04, 0.16, 0.04, 0.08, 0.06, 0.20, 0.06],
    "support": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.06, 0.04, 0.04],
    "confidence": [0.25, 0.33, 0.17, 0.5, 1.0, 0.125, 0.125, 0.5, 0.25, 0.5, 0.25, 0.667, 0.2],
    "lift": [4.167, 4.167, 4.167, 4.167, 6.25, 6.25, 3.125, 3.125, 6.25, 6.25, 4.167, 3.333, 3.333],
    "leverage": [0.0152, 0.0152, 0.0152, 0.0152, 0.0168, 0.0168, 0.0136, 0.0136, 0.0168, 0.0168, 0.0456, 0.028, 0.028],
    "conviction": [1.253, 1.38, 1.152, 1.76, float("inf"), 1.12, 1.097, 1.68, 1.28, 1.84, 1.253, 2.4, 1.175],
    "zhangs_metric": [0.826, 0.809, 0.863, 0.792, 0.857, 1.0, 0.81, 0.708, 0.913, 0.875, 1.0, 0.745, 0.875]
}

df_rules = pd.DataFrame(data)

# Convert antecedents and consequents into a matrix
matrix = df_rules.pivot(index='antecedents', columns='consequents', values='lift')

# Plot heatmap
st.subheader('Association Rules Heatmap')
fig = px.imshow(matrix, labels=dict(x="Consequents", y="Antecedents", color="Lift"),
                x=matrix.columns, y=matrix.index)
st.plotly_chart(fig, use_container_width=True)


#create pie charts for segment sales and category sales
chart1, chart2 = st.columns((2))
with chart1:
    st.subheader('Segment-Sales')
    fig = px.pie(filtered_df, values="Sales", names="Segment", template="plotly_dark", 
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(text=filtered_df["Segment"], textposition="inside")
    st.plotly_chart(fig, use_container_width=True)

with chart2:
    st.subheader('Category-Sales')
    fig = px.pie(filtered_df, values="Sales", names="Category", template="gridon", 
                 color_discrete_sequence=px.colors.qualitative.Set1)
    fig.update_traces(text=filtered_df["Category"], textposition="inside")
    st.plotly_chart(fig, use_container_width=True)


# Scatter plot for Sales vs Profit
data1 = px.scatter(filtered_df, x="Sales", y="Profit", size="Quantity")
data1['layout'].update(title="Relationship between Sales and Profits using Scatter Plot.",
                       titlefont=dict(size=20), xaxis=dict(title="Sales", titlefont=dict(size=19)),
                       yaxis=dict(title="Profit", titlefont=dict(size=19)))
st.plotly_chart(data1, use_container_width=True)

# Time Series Analysis
filtered_df["month_year"] = filtered_df["Order Date"].dt.to_period("M")
st.subheader('Time Series Analysis')
linechart = pd.DataFrame(filtered_df.groupby(filtered_df["month_year"].dt.strftime("%Y : %b"))["Sales"].sum()).reset_index()
fig2 = px.line(linechart, x="month_year", y="Sales", labels={"Sales": "Amount"}, height=500, width=1000, template="gridon")
st.plotly_chart(fig2, use_container_width=True)

with st.expander("View Data of TimeSeries:"):
    st.write(linechart.T.style.background_gradient(cmap="Blues"))
    csv = linechart.to_csv(index=False).encode("utf-8")
    st.download_button('Download Data', data=csv, file_name="TimeSeries.csv", mime='text/csv')

import plotly.figure_factory as ff
st.subheader(":point_right: Month wise Sub-Category Sales Summary")
with st.expander("Summary_Table"):
    df_sample = df[0:5][["Region", "State", "City", "Category", "Sales", "Profit", "Quantity"]]
    fig = ff.create_table(df_sample, colorscale="Cividis")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Month wise sub-Category Table")
    filtered_df["month"] = filtered_df["Order Date"].dt.month_name()
    sub_category_Year = pd.pivot_table(data=filtered_df, values="Sales", index=["Sub-Category"], columns="month")
    st.write(sub_category_Year.style.background_gradient(cmap="Blues"))

# View Summary
if st.button("View Summary"):
    st.subheader("Summary Table")
    st.write(filtered_df.groupby(['Category', 'Sub-Category']).agg({'Sales': 'sum', 'Quantity': 'sum'}).reset_index())

# Download Summary
if st.button("Download Summary"):
    csv = filtered_df.groupby(['Category', 'Sub-Category']).agg({'Sales': 'sum', 'Quantity': 'sum'}).reset_index().to_csv(index=False)
    st.download_button(label="Download Summary (CSV)", data=csv, file_name="summary.csv")
