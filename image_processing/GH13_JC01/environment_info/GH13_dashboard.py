# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 20:39:44 2023

@author: jcard
"""
import pandas as pd
import plotly.express as px
import datetime as dt
import plotly.io as pio
# libraries needed for the dashboard configuration
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc # webapp customization
from dash.dependencies import Input, Output

# Here I'm importing and cleaning my data. Retrieve environmental information using campbell library.
# Most likely we will need to edit the dates and variable names. 
df = pd.read_csv("C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/environment_info/Daily_Indoors_GH13.csv")
df= df.drop(df.columns[[1,2,3,4]],axis = 1)
df = df.rename(columns={'Datetime': 'TIMESTAMP'})
#df.info()
#print(df[:5])

df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
df = df.rename(columns={df.columns[0]: df.columns[0],
                        **{col: col[2:-1] for col in df.columns[1:]}})
#df['TIMESTAMP']
#print(df[:5])
# Create a dataframe organized in a long format to apply facet

value_col = list(df.columns[1:])

# reshape the dataframe from wide format to long format using pd.melt()

df_long = pd.melt(df, id_vars=["TIMESTAMP"], value_vars= value_col, var_name="column_name", value_name="value")

#fig = px.line(df_long, x = 'TIMESTAMP', y = "value", facet_col = "column_name", facet_col_wrap =2)
#pio.show(fig)
#df.loc[(df['TIMESTAMP'] >= '07/16/2022')& (df['TIMESTAMP'] < '8/16/2022')]

# Here I will build my components: 
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])  

title = dcc.Markdown(children = "# Environment Monitoring Dashbord" ,style={"textAlign": "center"})

time_series = dcc.Graph(figure = {},
                        style={'height': '1000px', 'width': '1000px'})

#dropdown = dcc.Dropdown(options=df_long["column_name"].unique(),
#                        style = {'width': "40%"})

date_selection = dcc.DatePickerRange(
                    min_date_allowed = df_long['TIMESTAMP'][1],
                    max_date_allowed = df_long['TIMESTAMP'].iloc[-1],
                    start_date = df_long['TIMESTAMP'][1],
                    end_date = df_long['TIMESTAMP'].iloc[-1])


# Customize our own layout: 
    
app.layout = dbc.Container([title,date_selection,time_series])
    
# Callbacks is what allows my app components to interact. 
@app.callback(
    Output(time_series, component_property="figure"),
    [Input(date_selection, component_property= "start_date"),
     Input(date_selection, component_property= "end_date")])
#     Input(dropdown, component_property="value")])

def update_graph(start_date,end_date):
   filtered_df = df_long.loc[(df_long['TIMESTAMP'] >= start_date)& (df_long['TIMESTAMP'] < end_date)]
   
   fig = px.line(filtered_df, x = 'TIMESTAMP', y = "value", facet_col = "column_name", facet_col_wrap =3)
   fig.update_layout(title = 'Environmental Data Experiment GH13JC01',
                     xaxis_title = 'Dates'
                     )
   fig.update_xaxes(type ="date",
                    tickformat = '%b %d')
   fig.update_traces(line_color='green')
   fig.update_yaxes(matches=None)
   fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])) # avoids column name to be displayed on top of each facet.
   
   return fig
    
if __name__ == '__main__':
    app.run_server(debug =True)

