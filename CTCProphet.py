# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:43:45 2021

@author: Connor.Bank
"""

#%%
import math
import fbprophet
import pandas as pd
import datetime as dt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter
from matplotlib import pyplot as plt
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from fbprophet.diagnostics import cross_validation
from datetime import date
from sklearn.metrics import mean_absolute_error
import pyodbc
import schedule
import time
import yagmail
import xlsxwriter

#%%        
#The whole program is defined as a function so that the scheduler can run it daily.
#For timeseries purposes we will be implementing Facebook's Prophet model, which is a 
#forecasting model desinged to be implemented in R/Python and then fine tuned by a data 
#analyst or scientist. facebook.github.io/prophet/ to read more about this awesome open
#source forecasting tool.
#Connect to the Scratch database in SQL Server so that we can store a query output as a 
#dataframe object for the model to take as input. Our Qurey is fairly simple, it uses
#the past few years of pipleine data for Clear to Close filtering down to only retail loans


path = "C:/Users/conno/Documents/ProphetTimeSeries/Electric_Production.csv"

#Using the Pandas library we assign the ouput of the query to a dataframe object
df = pd.read_csv(path)

#Next there is a little housekeeping/data cleaning that needs to happen to prepare
#the query output for the Prophet model.
#First, change Date column explicitly to 'datetime' format
df['DATE'] = df['DATE'].astype('datetime64[ns]')

#Prophet takes 2 very specific input columns, so here we create our columns 
#to the required names.
#create new ds column from Date
df['ds'] = df['DATE']
#create new y column from CTCPipe
df['y'] = df['Value']
#repurpose date column to index
df = df.set_index('DATE')


#Set some parameters for the plots that will be created down the line
plt.figure(figsize=(11,8))
ax = plt.subplot(111)

#Plot the entire timeseries output
df['Value'].plot(color='#334f8d', fontsize=11, zorder=2, ax=ax)

#Setting more parameters for a dual axis chart that will compare
#the Boxcox transformation with the original dataset.
#despine
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

#remove x-axis label
ax.set_xlabel('')

#switch off ticks
ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='on', left='off', right='off', labelleft='on')

#annotate
x_line_annotation = dt.datetime(2018, 5, 1)
x_text_annotation = dt.datetime(2018, 5, 5)

#get y-axis tick values
vals = ax.get_yticks()

#draw horizontal axis lines
for val in vals:
    ax.axhline(y=val, linestyle='dashed', alpha=0.3, color='#eeeeee', zorder=1)
    
#format y-axis label
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

#set y-axis label
ax.set_ylabel('Value', labelpad=20, weight='bold')

#set y-axis limit
ylim = ax.set_ylim(bottom=0)



#The dependent variable in our timeseries is not normally distributed
#so we will apply a Box-Cox transformation, which will help to normalize 
#the data so that we can run the data through our model.
#apply boxcox transform
df['y'], lam = boxcox(df['Value'])

#print lambda value, which is the power to which all data should be raised to 
#achieve normalization. This is not necessary but really cool to see as our data changes.
#The lambda value for this data set is not quite the PERFECT specimen, but falls well within
#the desired range to use Box-Cox for some good lift
print('Lambda: {}'.format(lam))


#create stacked plot comparison
ax = df[['Value', 'y']].plot(color='#334f8d', subplots=True, sharex=True, fontsize=11, legend=False, figsize=(11,12), title=['Untransformed', 'Box-Cox Transformed'])
for i, x in enumerate(ax):
    #despine
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)
    x.spines['left'].set_visible(False)
    x.spines['bottom'].set_visible(False)
    #remove axis label
    x.set_xlabel('')
    #switch off ticks
    x.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='on', left='off', right='off', labelleft='on')
    #format y-axis ticks
    vals = x.get_yticks()
    x.set_yticklabels(['{:,}'.format(int(y)) for y in vals])
    #draw horizontal axis lines
    for tick in vals:
        x.axhline(y=tick, linestyle='dashed', alpha=0.3, color='#eeeeee', zorder=1)
    #format y-axis label
    x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    #set y-axis limit
    if i == 0:
        x.set_ylim(bottom=0)

#create Prophet model object
m = fbprophet.Prophet(interval_width=0.95, changepoint_prior_scale=1)

#fit Prophet object with prepared dataframe
m.fit(df)

#create a dataframe with ds extending 'n' number of periods into the future.
#The model will make a prediciton for every data point in our timeseries, but
#we will also have it extend into the future to make a forecast
future = m.make_future_dataframe(periods=30)

#create forecast for 'n' periods
forecast = m.predict(future)

#Plot forecast with upper and lower bounds band
m.plot(forecast)

#Plot the overall trend as well as weekly/yearly trend
m.plot_components(forecast)



#apply inverse box cox transform to reset the data back to the expected values
forecast[['yhat','yhat_upper','yhat_lower']] = forecast[['yhat','yhat_upper','yhat_lower']].apply(lambda x: inv_boxcox(x, lam))
forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast['ds'] = pd.to_datetime(forecast['ds'])
forecast = forecast.set_index('ds')





#Now we start to create our output dataframe that will be written to an Excel file.
#Take the final 60 elements of the forecast, as that will be the past 30 days and the 30 days
#in the future
csvforecast = forecast[-60:]
csvforecast= csvforecast.rename(columns={"yhat": 'Prediction', 'yhat_lower': 'Prediction Lower Bound', 'yhat_upper': 'Prediction Upper Bound'})
csvforecast.index= csvforecast.index.rename('Date')

#Defining the Excel output columns
csvforecast['Actual']= df['Value']
csvforecast['Actual']= csvforecast['Actual'].fillna(0)
csvforecast['Delta'] = csvforecast['Actual']-(csvforecast['Prediction'].round())
csvforecast['Accuracy'] = 1-(abs(csvforecast['Actual']-csvforecast['Prediction']))/csvforecast[['Prediction', 'Actual']].max(axis=1)
csvforecast['Lower Bound Accuracy'] = 1-(abs(csvforecast['Actual']-csvforecast['Prediction Lower Bound']))/csvforecast[['Prediction Lower Bound', 'Actual']].max(axis=1)
csvforecast['Upper Bound Accuracy'] = 1-(abs(csvforecast['Actual']-csvforecast['Prediction Upper Bound']))/csvforecast[['Prediction Upper Bound', 'Actual']].max(axis=1)

#Rounding some output values for readabulity
csvforecast = csvforecast.round(4)
csvforecast['Prediction'] = csvforecast['Prediction'].round()
csvforecast['Prediction Upper Bound'] = csvforecast['Prediction Upper Bound'].round()
csvforecast['Prediction Lower Bound'] = csvforecast['Prediction Lower Bound'].round()

csvforecast = csvforecast[['Prediction', 'Actual', 'Delta', 'Accuracy', 'Prediction Lower Bound', 'Prediction Upper Bound', 'Lower Bound Accuracy', 'Upper Bound Accuracy']]
#print(csvforecast.tail())

#Load in our Excel writer so that we can format our output with conditional formatting
writer = pd.ExcelWriter('30dayforecast.xlsx', engine='xlsxwriter', datetime_format='mmm d yyyy')
pd.DataFrame(csvforecast.to_excel(writer, sheet_name='ValueForecast'))
workbook = writer.book
worksheet = writer.sheets['ValueForecast']
worksheet.conditional_format('E2:E31', {'type': '3_color_scale'})
worksheet.conditional_format('H2:I31', {'type': '3_color_scale'})
dashes = ('-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-')
worksheet.write_column('E32:E61', dashes)
worksheet.write_column('H32:H61', dashes)
worksheet.write_column('I32:I61', dashes)
worksheet.write('K2', 'Avg Acc')
worksheet.write('L2', 'Avg Err')
worksheet.write_formula('K3', '=AVERAGE(E2:E15)')
worksheet.write_array_formula('L3', '{=AVERAGE(ABS(D2:D15))}')
worksheet.write('M2', 'Over Predict')
worksheet.write('N2', 'Under Predict')
worksheet.write_formula('M3', '=COUNTIF(D2:D17, "<0")')
worksheet.write_formula('N3', '=15-M3')
header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top'})
for col_num, value in enumerate(csvforecast.columns.values):
    worksheet.write(0, col_num + 1, value, header_format)

worksheet.set_column('A:L', 11)
pct_format = workbook.add_format({'num_format': '##.00%'})
worksheet.set_column('E2:E32', 10, pct_format)
worksheet.set_column('H2:I32', 10, pct_format)
worksheet.set_column('K3:K4', 10, pct_format)
worksheet.set_column('M2:N4', 15)





   
writer.save()
