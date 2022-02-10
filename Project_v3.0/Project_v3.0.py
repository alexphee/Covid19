
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import plotly.express as px
import plotly.graph_objects as go

plt = matplotlib.pyplot
plt2 = matplotlib.pyplot

#################
# Global status #
#################
#κατάληξη url σε array
plt2.rcParams["figure.figsize"] = (14,8)
countries = ['country/greece', 'country/brazil', 'country/china', 'country/uk', 'country/italy', '']
k=-1
#ξεχωριστό array για το memo
countries_memo = ['Greece', 'Brazil', 'China', 'UK', 'Italy', 'World']

#γραφική αναπαράσταση για κάθε χώρα στο ίδιο γράφημα
for country in countries:
    k=k+1
    data = requests.get("https://www.worldometers.info/coronavirus/"+ country) # για να εμφανίζεται και το World
    data = data.text
    tot = data[data.find("data: [")+len("data: ["):data.find("]", data.find("data: [")+len("data: ["))]
    tot = [float(i) for i in tot.split(',')]
    dates = data [data.find("categories: [")+len("categories: [") : data.find("]" , data.find("categories: [")+len("categories: ["))]
    dates = [date for date in dates.split(",")]
    plt2.plot( tot ,  label = countries_memo[k])
plt2.legend()
plt2.grid(which = "both")
plt2.yscale("log")          #log scale για να εμφανίζονται ομοιόμορφα οι καμπύλες
plt2.xticks(rotation=90)
plt2.show()

#################
# Greece status #
#################
df = pd.read_csv("data/covidGR.csv")
fig = px.line(df, x = 'date', y = 'total_cases', title='Κρούσματα στην Ελλάδα (έως 18/04/2021)')
fig.write_html('Greek Cases.html', auto_open=True)

##################
# Global heatmap #
##################
df = pd.read_csv("data/time_series_covid19_confirmed_global.csv")
df = df.rename(columns= {"Country/Region" : "Country", "Province/State": "Province"})
total_list = df.groupby('Country')['2/27/21'].sum().tolist()

country_list = df["Country"].tolist()
country_set = set(country_list)
country_list = list(country_set)
country_list.sort()

new_df = pd.DataFrame(list(zip(country_list, total_list)), 
               columns =['Country', 'Total_Cases'])

colors = ["#F9F9F5", "#FAFAE6", "#FCFCCB", "#FCFCAE",  "#FCF1AE", "#FCEA7D", "#FCD97D",
          "#FCCE7D", "#FCC07D", "#FEB562", "#F9A648",  "#F98E48", "#FD8739", "#FE7519",
          "#FE5E19", "#FA520A", "#FA2B0A", "#9B1803",  "#861604", "#651104", "#570303",]

fig = go.Figure(data=go.Choropleth(
    locationmode = "country names",
    locations = new_df['Country'],
    z = new_df['Total_Cases'],
    text = new_df['Total_Cases'],
    colorscale = colors,
    autocolorscale=False,
    reversescale=False,
    colorbar_title = 'Επιβεβαιωμένα κρούσματα Covid-19',
))

fig.update_layout(
    title_text='Επιβεβαιωμένα κρούσματα Covid-19',
    geo=dict(
        showcoastlines=True,
    ),
)

fig.write_html('World Map.html', auto_open=True)

##############################
# Machine Learning Algorithm #
##############################
plt.rcParams["figure.figsize"] = (14,8)

data = pd.read_csv("data/total_cases.csv", sep = ',') #διαβασμα αρχειου csv, χωρίζόνται με ,
data = data[['date','World']]                           #διάβασμα των 2 στηλών
print('-'*30)                                           #τυπώνει πληροφορίες
print('mai21061')
print('-'*30)
print(data.head(500))

print('-'*30)
print('PREPARING DATA')                                 #προετοιμασία δεδομένων
print('-'*30)
x = np.array(data['date']).reshape(-1, 1)               #μετατροπή σε numpy arrays
y = np.array(data['World']).reshape(-1, 1)              #reshape ώστε να έχουν ίδιο μέγεθος
plt.plot(y,'-b.')                                       #γράφημα με μπλε χρώμα

polyFeat = PolynomialFeatures(degree=5)                 #καθορισμός του βαθμού του πολυωνύμου (πχ degree=2 --> x, x^2)
x = polyFeat.fit_transform(x)                           #κάθε βαθμός πρέπει να έχει τη στήλη του
print(x)
print('-'*30)
print('TRAINING MODEL')                                 #προετοιμασία γραμμικού μοντέλου
print('-'*30)
model = linear_model.LinearRegression()
model.fit(x,y)                                          #προσαρμογή μοντέλου στα πραγματικά δεδομένα
accuracy = model.score(x,y)                             #ακρίβεια προσαρμογής
print(f'Accuracy:{round(accuracy*100,5)}%')
y0 = model.predict(x)                                   #μέθοδος πρόβλεψης
plt.plot(y0,'--r')                                      #γραφική αναπαράσταση προσαρμογής, κόκκινη διακεκομμένη
plt.show()

days = 2
print('-'*30)
print('PREDICTING GROWTH')                              #πρόβλεψη μοντέλου για +2 μέρες
print('-'*30)
print(f'Prediction - Cases after {days} days:',end='')
print(round(int(model.predict(polyFeat.fit_transform([[484+days]])))/1000000,2),'Million cases')
print('-'*30)