import pandas as pd
from scipy.stats.mstats import winsorize
from scipy.stats import boxcox
from sklearn.preprocessing import LabelEncoder
from preprocessing_functions import get_season, get_period_of_day, \
    get_duration, get_duration_hours, get_mapped_value


df = pd.read_excel("data/df.xlsx")

df = df.dropna()

df["Airline"] = df["Airline"].str.lower()

df['month'] = pd.DatetimeIndex(df['Date_of_Journey']).month  # month
df['day'] = pd.DatetimeIndex(df['Date_of_Journey']).day  # day
df['season'] = df['month'].apply(get_season)  # season
df['Dep_Time_Period'] = df['Dep_Time'].apply(get_period_of_day)  # period of day (departure time)
df['Arrival_Time_Period'] = df['Arrival_Time'].apply(get_period_of_day)  # period of day (arrival time)
df['Duration_Minutes'] = df['Duration'].apply(get_duration)  # duration of a flight
df['Duration_Hours'] = df['Duration'].apply(get_duration_hours)  # duration of a flight in hours

df["Total_Stops"] = df["Total_Stops"]\
    .map({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})  # total stops in a period of flight
df["season_enc"] = df["season"]\
    .map({'spring': '0', 'summer': '1', 'fall': '2', 'winter': '3'})  # season of flight
df["Arrival_Time_Period_enc"] = df["Arrival_Time_Period"]\
    .map({'Early Morning': 0, 'Morning': 1, 'Noon': 2, 'Eve': 3, 'Night': 4, 'Late Night': 5})  # name of day periods
df["Dep_Time_Period_enc"] = df["Dep_Time_Period"]\
    .map({'Early Morning': 0, 'Morning': 1, 'Noon': 2, 'Eve': 3, 'Night': 4, 'Late Night': 5})  # name of day periods
df = pd.get_dummies(df, columns=['Additional Info'])  # creating columns for each additional info

df_EDA = df.copy()

df["Airline_group"] = df["Airline"].apply(get_mapped_value)

df["Price_wins"] = winsorize(df["Price"], limits=0.01)
df["Price_log"], param_1 = boxcox(df["Price_wins"])  # param_1 for inverse transformation

df["Duration_Minutes_wins"] = winsorize(df["Duration_Minutes"], limits=0.01)

le_1 = LabelEncoder()
le_2 = LabelEncoder()

df["Airline"] = le_1.fit_transform(df["Airline"])
df["Airline_group"] = le_2.fit_transform(df["Airline_group"])

keys_1 = le_1.classes_
values_1 = le_1.transform(le_1.classes_)
dictionary_airlines = dict(zip(keys_1, values_1))

keys_1 = le_2.classes_
values_1 = le_2.transform(le_2.classes_)
dictionary_groups = dict(zip(keys_1, values_1))

df_selected = df.copy()
df_selected = df_selected.drop(["Additional Info_No info", "Additional Info_No Info", 'Route', 'Dep_Time',
                                'Arrival_Time', 'Duration', 'Date_of_Journey', 'Source', 'Destination',
                                'Duration_Minutes', 'Price', 'Price_wins', 'season', 'Dep_Time_Period',
                                'Arrival_Time_Period', 'Additional Info_2 Long layover',
                                'Additional Info_1 Short layover', 'Additional Info_Red-eye flight',
                                'Duration_Hours', 'Arrival_Time_Period_enc'], axis=1)

print(df_selected.head())