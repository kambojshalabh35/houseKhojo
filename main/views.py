from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder  
from sklearn.linear_model import LinearRegression
import math

fs = FileSystemStorage()
data = fs.open('housing.csv')

data = pd.read_csv(data)
data = data.drop(['median_income'], axis=1)
data = data.drop(['population'], axis=1)

data.dropna(inplace=True)

data['total_rooms'] = np.log(data['total_rooms']+ 1)
data['total_bedrooms'] = np.log(data['total_bedrooms']+ 1)
data['households'] = np.log(data['households']+ 1)

data['bedroom_ratio']= data['total_bedrooms']/data['total_rooms']
data['household_rooms']= data['total_rooms']/data['households']

data = data.join(pd.get_dummies(data.ocean_proximity)).drop(['ocean_proximity'], axis=1)
data = data.drop(['<1H OCEAN'], axis=1)

x=data.drop(['median_house_value'], axis=1)
y=data['median_house_value']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

train_data=x_train.join(y_train)


reg= LinearRegression()

reg.fit(x_train, y_train)

# forest = RandomForestRegressor()

# forest.fit(x_train, y_train)

# Create your views here.
def home(request):
    if(request.method == 'POST'):
        Longitude = int(request.POST['Longitude'])
        Latitude = int(request.POST['Latitude'])
        Age_of_House = int(request.POST['Age_of_House'])
        Total_Rooms = request.POST['Total_Rooms']
        Number_of_Bedrooms = request.POST['Number_of_Bedrooms']
        Households = request.POST['Households']

        Inland = bool(request.POST['Inland_v'] == 'True')
        Island = bool(request.POST['Island_v'] == 'True')
        Near_the_Bay = bool(request.POST['Near_the_Bay_v'] == 'True')
        Near_the_Ocean = bool(request.POST['Near_the_Ocean_v'] == 'True')

        Total_Rooms= np.log(int(Total_Rooms)+ 1)
        Number_of_Bedrooms= np.log(int(Number_of_Bedrooms)+ 1)
        Households= np.log(int(Households)+ 1)

        bedroom_ratio= Number_of_Bedrooms/Total_Rooms
        household_rooms= Total_Rooms/Households

        inp = [Longitude, Latitude, Age_of_House, Total_Rooms, Number_of_Bedrooms, Households, bedroom_ratio, household_rooms, Inland, Island, Near_the_Bay, Near_the_Ocean]

        output = reg.predict([inp])[0]

        if(output < 0):
            output = -output

        output = math.ceil(output*100)/100

        
        return render(request, 'output.html', {'output': output * 60, 'heatmap':train_data.corr(), "longitudes":train_data['longitude'].tolist(), "latitudes":train_data['latitude'].tolist(), "isHome": False})

    return render(request, 'home.html', {'isHome': True})

def contact(request):
    return render(request, 'contact.html')

def about(request):
    return render(request, 'about.html')