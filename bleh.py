# importing
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statistics import mean

#the plan:
#read the master data file as a csv w/pandas
#target the variable
#use techniques to preprocess data
#use scikit larn to randomize splitting into testing and training data
#train the model
#graph the thing

def readFile(file):
	df = pd.read_csv(file, sep='\t')
	return df

def printDF(datfra):
	print("getting the info:")
	print(".head")
	print(datfra.head())
	print(".info")
	print (datfra.info())
	print(".tail")
	print(datfra.tail())

def linReg(X_train, y_train, X_test, y_test): 
	regr = linear_model.LinearRegression()
	regr.fit(X_train, y_train)
	y_pred = regr.predict(X_test)
	print ("The thing has been trained")
	#print('Coefficients: \n', regr.coef_)
	print("Mean squared error: %.2f"
		% mean_squared_error(y_test, y_pred))
	print("OTHER Mean squared error:",  mean_squared_error(y_test, y_pred))
	return (y_pred)

def barplot(x_val, y_val, xaxis, yaxis, title):
	plt.bar(x_val, y_val, align='center', alpha=0.5, width = .02)
	#how do I do xticks and yticks
	plt.ylabel(xaxis)
	plt.xlabel(yaxis)
	plt.title(title)
	plt.show()

def scatterplot(x_val, y_val, xaxis, yaxis, title):
	plt.scatter(x_val, y_val)
	plt.xlabel(xaxis)
	plt.ylabel(yaxis)
	plt.title(title)

	plt.xlim(xmin = 0, xmax =6) #setting the same axes scaling for both sides
	plt.ylim(ymin=0, ymax=6)
	
	plt.show()

def sidebysideplot(df): #takes the DF and the index that you want to set it to, then sorts it and plots the two things together.
	df.sort_index(inplace=True)
	df.plot.bar()
	plt.legend()
	plt.show()
	return df

def errorpercentline(df):
	df.sort_index(inplace=True)
	df.reset_index(inplace=True)
	early = {}
	currentindex =1
	total_predicted = 0
	total_actual = 0
	a = np.zeros(shape = (len(df), 1))
	for index, row in df.iterrows():
		item =abs((row['predicted'] - row['actual'])/row['actual'])*100
		a[index, 0] = item
	df['error'] = a
	print ('mean error percent', mean(df['error']))
	df.set_index('yid', inplace=True)
	df['error'].plot.bar()
	plt.legend()
	plt.show()
	return df


def smalldf(df): #takes the df with the multiple indexes n stuff and avges the values so it's easier to graph
	df.sort_index(inplace=True)
	a = np.zeros(shape = (38, 2))
	for i in range(38):
		#print (df.loc[i])
		#print ('predicted mean', mean(df.loc[i, 'predicted']))
		mean_pred = mean(df.loc[i, 'predicted'])
		#print ('actual mean', mean(df.loc[i, 'actual']))
		mean_actual = mean(df.loc[i, 'actual'])
		a[i, 0] = mean_pred
		a[i, 1] = mean_actual
	newdf = pd.DataFrame(a)
	newnames = {0:'predicted pwr', 1:'actual pwr'}
	newdf.rename(columns=newnames, inplace=True) 
	return newdf
	

def datatodictionary(df): #for the purpose of renaming the workload id to names
	namesdict = df[0:39]['Workload_Name']
	namesdict = namesdict.to_dict()
	print (type(namesdict))
	return namesdict


def allthegraphs(graph1, graph2, graph3, graph4): #put the graphs together, gotta b 4
	print ("This function shall be made someday")

df = readFile('entiredataset.csv')
df_target = df['Power_A15'] #, 'Power_A7' took out power a7 for now
#printDF(df)
df_train = df.drop(['Unnamed: 0','Unnamed:_0', 'Workload_Name','Core_Mask',
	'Power_A7','Power_A15','Status','Power_A7', 
	'Core_4_Predicted_Dynamic_Power','Core_5_Predicted_Dynamic_Power', 
	'Core_6_Predicted_Dynamic_Power','Core_7_Predicted_Dynamic_Power', 
	'Summed_A15_Cores_Dynamic_Power', 'Switching_Dynamic_Power_A15',
	'Total_Static_Power_A15','Total_Power_Summed_A15', 
	'Core_4_Static_Power', 'Core_5_Static_Power', 'Core_6_Static_Power',
	'Core_7_Static_Power','L2_and_Background_Static_Power', 
	'Core_4_Static_Power_CC','Core_7_Static_Power_CC',
	'Total_Static_Power_A15_CC'], axis=1)

X_train, X_test, y_train, y_test= train_test_split(df_train, df_target, test_size=0.25, random_state=42)
y_pred = linReg(X_train, y_train, X_test, y_test)
barplot(y_pred, y_test, 'y_pred', 'y_test', 'the graph')
scatterplot(y_pred, y_test, 'y_pred', 'y_test', 'second graph')

newdf = pd.DataFrame({'actual':y_test.values, 'predicted':y_pred, 'yid':X_test['Workload_ID']}) 
newdf.set_index('yid', inplace=True)
sidebysideplot(newdf)

errorpercentline(newdf)
shortdf = smalldf(newdf)


namesdict = datatodictionary(df)
shortdf.reset_index(inplace=True)
shortdf.replace(namesdict, inplace = True)
print (shortdf)

#shortdf.replace(namesdict, inplace = True)
#print (shortdf)


shortdf.plot.bar(x='index')
plt.title('Predicted vs Actual: Power A15')
plt.show()


'''
'zipped to make a bar graph'
zippy = dict(zip(y_pred[0:6], y_test[0:6]))
plt.xticks(range(len(zippy)), zippy.keys())
plt.bar(range(len(zippy)), zippy.values(), align='center', width = 0.2)
plt.show()
'''


'''
data = pd.DataFrame(df_main)

print (data['Core Count Both'])
#print (data.iloc[4])

plt.plot(data['EPH_0x14'])
plt.title('Core Count Both data')
plt.xlabel('rows?')
plt.ylabel('value')
plt.legend()
plt.show()


D = {'Label0':26, 'Label1': 17, 'Label2':30}
plt.xticks(range(len(D)), D.keys())
plt.bar(range(len(D)), D.values(), align='center')
plt.show()



ax = plt.subplot(111)
x = [1, 2, 3, 4]
ax.bar(x-0.2, y_pred,width=0.2,color='b',align='center')
ax.bar(x, y_test,width=0.2,color='g',align='center')
plt.show()
'''