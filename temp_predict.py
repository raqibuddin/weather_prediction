import requests  #helps us to fetch data from API
import pandas as pd #for analysis
import numpy as np #for numerical operations
from sklearn.model_selection import train_test_split  #to split data into training and testing sets
from sklearn.preprocessing import LabelEncoder    #to convert catogerical data into numerical valuse
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor #models for classification and regression tasks
from sklearn.metrics import mean_squared_error  #to measure the accuracy
from datetime import datetime,timedelta #to handle date and ime
import pytz

def enter_api_key():
  API_KEY = None
  API_KEY = input("Enter your OpenWeatherMap API key: ").strip()

  test_url= f"https://api.openweathermap.org/data/2.5/weather?q=London&appid={API_KEY}"
  response = requests.get(test_url)
  if response.status_code != 200:
    print("Enter valid API_KEY")
    return enter_api_key()
  return API_KEY

def get_current_weather(city=None, lat=None, lon=None, API_KEY=None):

  BASE_URL='https://api.openweathermap.org/data/2.5/' #base url for making API request

  if city:
        url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
  else :
        url = f"{BASE_URL}weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

  response = requests.get(url) #send the get request to API
  data = response.json() #parse the response as JSON

  return {
      'city':data['name'],
      'current_temp': round(data['main']['temp']),
      'feels_like': round(data['main']['feels_like']),
      'temp_min': round(data['main']['temp_min']),
      'temp_max': round(data['main']['temp_max']),
      'humidity': round(data['main']['humidity']),
      'description':data['weather'][0]['description'],
      'country':data['sys']['country'],
      'wind_gust_dir':data['wind']['deg'],
      'pressure':data['main']['pressure'],
      'Wind_Gust_Speed':data['wind']['speed']
  }


def read_historical_data(filename):
  df = pd.read_csv(filename)  #read file into dataframe
  df=df.dropna() #remove rows with missing values
  df=df.drop_duplicates() #remove duplicate rows
  return df


def prepare_data(data):
  le =LabelEncoder() #create a labelencoder
  data['WindGustDir'] = le.fit_transform(data['WindGustDir']) #encode categorical data
  data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

  x=data[['MinTemp','MaxTemp','WindGustDir','WindGustSpeed','Humidity','Pressure','Temp']] #feature variables
  y=data['RainTomorrow'] #target variable

  return x,y,le


def train_rain_model(x,y):
  x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42) #split data into training and testing sets
  model = RandomForestClassifier(n_estimators=100, random_state=42) #create a random forest classifier
  model.fit(x_train, y_train) #train the model on the training data

  y_pred = model.predict(x_test) #make predictions on the testing data
  accuracy = mean_squared_error(y_test,y_pred) #calculate the accuracy of the model
  print("Mean squuared error value:",accuracy) #print the accuracy of the model

  return model


def prepare_regression_data(data, feature):
  x_list,y_list= [] , []  #data set to added and initialize list for feature and target values
  for i in range(len(data)-1):
    x_list.append(data[feature].iloc[i])
    y_list.append(data[feature].iloc[i+1])

    x=np.array(x_list).reshape(-1,1)
    y=np.array(y_list)
  return x,y


def train_regression_model(x,y):
  model=RandomForestRegressor(n_estimators=100,random_state=42) #create a random forest regressor
  model.fit(x,y) #train the model on the data

  return model

def predict_future(model, current_value):
  predictions = []
  last_known_value = current_value

  for i in range(5):
    next_value = model.predict(np.array([[last_known_value]]))
    last_known_value = next_value[0]
    predictions.append(last_known_value)

  return predictions

def weather_view():

    API_KEY=None
    API_KEY=enter_api_key()
    if not API_KEY:
      print("Please provide your OpenWeatherMap API key in the API_KEY variable.")
      return enter_api_key()

    # Ask user how they want to search
    method = input("Search by city name or by coordinates?(Enter 'city' or 'coord'): ").strip().lower()

    city = None
    lat = None
    lon = None

    if method == 'city':
        city = input("Enter city name: ").strip()
    elif method == 'coord':
      lat = float(input("Enter latitude: ").strip())
      lon = float(input("Enter longitude: ").strip())

    # Fetch current weather using either city or lat/lon
    current_weather= get_current_weather(city=city, lat=lat, lon=lon, API_KEY=API_KEY)

    if not current_weather:
        print(f"Could not retrieve weather data. Please check your input and API key.")
        return


  #load historical data
    historical_data=read_historical_data('weather.csv')

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
      'Temp':current_weather['current_temp']
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

  #display results
    print(f"City: {city} ,{current_weather['country']}")
    print(f"Current Temperature: {current_weather['current_temp']}°C")
    print(f"Feels Like:{current_weather['feels_like']} °C")
    print(f"Min temp:{current_weather['temp_min']} °C")
    print(f"Max temp:{current_weather['temp_max']} °C")
    print(f"Humidity: {current_weather['humidity']} %")
    print(f"Weather prediction: {current_weather['description']}")
    print(f"Rain pridiction: {'Yes' if rain_prediction else 'No'}")

    print("\nFuture Temperature predictions:")

    for time, temp in zip(future_times,future_temp):
      print(f"{time}: {round(temp,1)} °C")

    print("\nFuture Humidity predictions:")

    for time, humidity in zip(future_times,future_humidity):
      print(f"{time}: {round(humidity,1)} %")

weather_view()