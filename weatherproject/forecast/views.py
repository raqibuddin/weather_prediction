from django.shortcuts import render, redirect

# Create your views here.
import requests  #helps us to fetch data from API
import pandas as pd #for analysis
import numpy as np #for numerical operations
from sklearn.model_selection import train_test_split  #to split data into training and testing sets
from sklearn.preprocessing import LabelEncoder    #to convert catogerical data into numerical valuse
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor #models for classification and regression tasks
from sklearn.metrics import mean_squared_error  #to measure the accuracy
from datetime import datetime,timedelta #to handle date and ime
import pytz 
import os


def enter_api_key(request):
  error=None
  if request.method == "POST":
    API_KEY = request.POST.get("api_key","").strip()

    if not API_KEY:
       return render(request, "enter_api_key.html", {"error": "API key is required"})

    test_url= f"https://api.openweathermap.org/data/2.5/weather?q=London&appid={API_KEY}"
    response = requests.get(test_url)

    if response.status_code == 200:
      request.session["api_key"]=API_KEY
      return redirect("Weather View")
    else:
      if "api_key" in request.session:
                del request.session["api_key"]
      error=" Invalid API key or the key is not been activated. Please try again."
  
  return render(request, "enter_api_key.html",{"error": error})


# 1.Fetch current Weather data
def get_current_weather(city=None, lat=None, lon=None, API_KEY=None):
    BASE_URL = 'https://api.openweathermap.org/data/2.5/'
    try:
        if city:
            url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
        else:
            url = f"{BASE_URL}weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

        response = requests.get(url)
        data = response.json()  # parse JSON before raising error

        #print("Lat:", lat, "Lon:", lon)

        # ✅ Check if API responded with an error
        if str(data.get("cod")) != "200":
            if str(data.get("cod")) == "404":
                return {"error_message": "City not found. Please check the spelling and try again."}
            elif str(data.get("cod")) == "401":
               return {"error_message": "Invalid API key. Please check your API key and try again."}
            else:
                return {"error_message": f"Weather service error: {data.get('message', 'Unknown error')}."}

        # ✅ If everything is good, return weather data
        return {
            'city': data['name'],
            'current_temp': round(data['main']['temp']),
            'feels_like': round(data['main']['feels_like']),
            'temp_min': round(data['main']['temp_min']),
            'temp_max': round(data['main']['temp_max']),
            'humidity': round(data['main']['humidity']),
            'sea_level': data['main'].get('sea_level', 'N/A'),
            'description': data['weather'][0]['description'],
            'country': data['sys']['country'],
            'wind_gust_dir': data['wind']['deg'],
            'pressure': data['main']['pressure'],
            'Wind_Gust_Speed': data['wind']['speed'],
            'clouds': data['clouds']['all'],
            'visibility': data['visibility'],
        }

    except requests.exceptions.RequestException:
        return {"error_message": "Unable to connect to weather service. Please try again later."}
    except Exception:
        return {"error_message": "Unexpected error while fetching weather data."}


# 2.Read Historical Data
def read_historical_data(filename):
  df = pd.read_csv(filename)  #read file into dataframe
  df=df.dropna() #remove rows with missing values
  df=df.drop_duplicates() #remove duplicate rows
  return df


# 3.Prepare data for training
def prepare_data(data):
  le =LabelEncoder() #create a labelencoder
  data['WindGustDir'] = le.fit_transform(data['WindGustDir']) #encode categorical data
  data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

  x=data[['MinTemp','MaxTemp','WindGustDir','WindGustSpeed','Humidity','Pressure','Temp']] #feature variables
  y=data['RainTomorrow'] #target variable

  return x,y,le


# 4.train Rain Prediction model
def train_rain_model(x,y):
  x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42) #split data into training and testing sets
  model = RandomForestClassifier(n_estimators=100, random_state=42) #create a random forest classifier
  model.fit(x_train, y_train) #train the model on the training data

  y_pred = model.predict(x_test) #make predictions on the testing data
  accuracy = mean_squared_error(y_test,y_pred) #calculate the accuracy of the model
  print("Mean squuared error value:",accuracy) #print the accuracy of the model

  return model


# 5.prepare regression data
def prepare_regression_data(data, feature):
  x_list,y_list= [] , []  #data set to added and initialize list for feature and target values
  for i in range(len(data)-1):
    x_list.append(data[feature].iloc[i])
    y_list.append(data[feature].iloc[i+1])

    x=np.array(x_list).reshape(-1,1)
    y=np.array(y_list)
  return x,y


# 6.Train regression data
def train_regression_model(x,y):
  model=RandomForestRegressor(n_estimators=100,random_state=42) #create a random forest regressor
  model.fit(x,y) #train the model on the data

  return model


# 7.predicting future
def predict_future(model, current_value):
  predictions = []
  last_known_value = current_value

  for i in range(5):
    next_value = model.predict(np.array([[last_known_value]]))
    last_known_value = next_value[0]
    predictions.append(last_known_value)

  return predictions


# 8.Weather Analysis Function
def weather_view(request):
    API_KEY = request.session.get("api_key")
    if not API_KEY:
        return redirect("enter_api_key")
    
    test_url = f"https://api.openweathermap.org/data/2.5/weather?q=Delhi&appid={API_KEY}&units=metric"
    response = requests.get(test_url)

    if response.status_code != 200:
        # ❌ invalid / expired / not activated key → clear session and redirect back
        request.session.pop("api_key", None)
        return redirect("enter_api_key")

    if request.method == 'POST':
        option = request.POST.get("option")
        city = None
        lat = None
        lon = None

        if option == "_city_":
            city = request.POST.get("city", "").strip()
            if not city:
                return render(request, 'weather.html', {"error_message": "Please enter a city name."})

        elif option == "_coords_":
            lat = request.POST.get("latitude", "").strip()
            lon = request.POST.get("longitude", "").strip()
            if not lat or not lon:
                return render(request, 'weather.html', {"error_message": "Please enter both latitude and longitude."})

        # Fetch current weather
        current_weather = get_current_weather(city=city, lat=lat, lon=lon, API_KEY=API_KEY)
        if "error_message" in current_weather:
          return render(request, 'weather.html', {"error_message": current_weather["error_message"]})

        #load historical data
        csv_path = os.path.join('S:\\weather_prediction\\weather.csv')
        historical_data=read_historical_data(csv_path)

        #prepare and train rain prediction model
        x,y,le=prepare_data(historical_data)

        rain_model=train_rain_model(x,y)

        #map wind direction to campass points
        wind_deg=current_weather['wind_gust_dir'] % 360
        compass_points=[('N', 0, 11.25),('NNE', 11.5, 33.75),('NE', 33.75, 56.25),
                  ('ENE', 56.25, 78.75),('E', 78.75 , 101.25),('ESE', 101.25, 123.75),
                  ('SE', 123.75 , 146.25),('SSE', 146.25, 168.75),('S', 168.75, 191.25),
                  ('SSW', 191.25, 213.75),('SW', 213.75, 236.25),('WSW', 236.25, 258.75),
                  ('W', 258.75, 281.25),('WNW', 281.25, 303.75),('NW', 303.75, 326.25),
                  ('NNW', 326.25, 348.75)]

        compass_direction=next(point for point,start,end in compass_points if start<=wind_deg< end)

        compass_direction_encoded=le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

        current_data={
        'MinTemp':current_weather['temp_min'],
        'MaxTemp':current_weather['temp_max'],
        'WindGustDir':compass_direction_encoded,
        'WindGustSpeed':current_weather['Wind_Gust_Speed'],
        'Humidity':current_weather['humidity'],
        'Pressure':current_weather['pressure'],
        'Temp':current_weather['current_temp'],
        #'sea_level':current_weather['sea_level'],
        #'Rain':current_weather['Rain'],
        }

        current_df=pd.DataFrame([current_data])

        # rain prediction
        rain_prediction=rain_model.predict(current_df)[0]

        #pepare regression model for temp and humidity
        x_temp,y_temp=prepare_regression_data(historical_data,'Temp')

        x_hum,y_hum=prepare_regression_data(historical_data,'Humidity')

        temp_model=train_regression_model(x_temp,y_temp)

        hum_model=train_regression_model(x_hum,y_hum)

        #predict future temp and humidity
        future_temp=predict_future(temp_model,current_weather['current_temp'])

        future_humidity=predict_future(hum_model,current_weather['humidity'])

        #prepare time for future prediction
        timezone=pytz.timezone('Asia/Kolkata')
        now=datetime.now(timezone)
        next_hour=now+timedelta(hours=1)
        next_hour=next_hour.replace(minute=0,second=0,microsecond=0)

        future_times=[(next_hour+timedelta(hours=i)).strftime("%H:00") for i in range(5)]

        # store each value separately

        time1,time2,time3,time4,time5 = future_times
        temp1,temp2,temp3,temp4,temp5 = future_temp
        hum1,hum2,hum3,hum4,hum5 = future_humidity
        
        #pass data to template
        context = {
           'location': city,
           'current_temp': current_weather['current_temp'],
           'MinTemp': current_weather['temp_min'],
           'MaxTemp': current_weather['temp_max'],
           'humidity': current_weather['humidity'],
           'clouds': current_weather['clouds'],
           'description': current_weather['description'],
           'city': current_weather['city'],
           'country': current_weather['country'],
           'feels_like':current_weather['feels_like'],
           'sea_level':current_weather['sea_level'],
           #'rain':current_weather['Rain'],

           'time': datetime.now(),
           'date': datetime.now().strftime("%B %d, %Y"),

           'wind': current_weather['Wind_Gust_Speed'],
           'pressure': current_weather['pressure'],
           'visibility': current_weather['visibility'],

            'time1': time1,
            'time2': time2,
            'time3': time3,
            'time4': time4,
            'time5': time5,

            'temp1': f"{round(temp1,1)}",
            'temp2': f"{round(temp2,1)}",
            'temp3': f"{round(temp3,1)}",
            'temp4': f"{round(temp4,1)}",
            'temp5': f"{round(temp5,1)}",

            'hum1': f"{round(hum1,1)}",
            'hum2': f"{round(hum2,1)}",
            'hum3': f"{round(hum3,1)}",
            'hum4': f"{round(hum4,1)}",
            'hum5': f"{round(hum5,1)}",   
          }
        
        if "error" in current_weather:
            # API or data error → show error message in template
          return render(request, 'weather.html', {"error": current_weather["error"]})
        return render(request, 'weather.html', context)
    
    return render(request, 'weather.html')