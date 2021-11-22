import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics


raw_data = pd.read_csv('weatherHistory.csv')
data = raw_data.head(20)
humidity = data.Humidity
temperature = data['Temperature (C)']
apparent_temperature= data['Apparent Temperature (C)']
pressure = data['Pressure (millibars)']

print(pd.DataFrame({'T':  temperature, 'A T': apparent_temperature}))

def linear_regression_with_chart(first_column, second_column, show_linear_regression = False, chart_title = 'Chart', chart_x_label='X', chart_y_label='Y',):
    linear_reg = LinearRegression()
    x = first_column.values.reshape(-1,1)
    y = second_column.values.reshape(-1,1)
    linear_reg.fit(x,y)
    prediction = linear_reg.predict(x)
    df = pd.DataFrame({'X':  first_column, 'Y': second_column})
    flat_prediction = [item for sublist in prediction for item in sublist]
    df_pr = pd.DataFrame({'Actual Value': second_column, 'Prediction': flat_prediction})
    print('Interception', linear_reg.intercept_)
    print(df_pr)
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(second_column, flat_prediction)))
    # Grafica
    plt.scatter(first_column, second_column)
    if(show_linear_regression):
        plt.plot(first_column, flat_prediction, c='red')
    plt.title(chart_title)
    plt.xlabel(chart_x_label)
    plt.ylabel(chart_y_label)
    plt.show()

def save_to_file(file_name, _data):
    f = open(f"{file_name}.txt", "w")
    f.write(_data)
    f.close()

save_to_file('correlation', raw_data.corr().to_string())
linear_regression_with_chart(humidity, temperature, False, 'Humidity and Temperature Relationship', 'Humidity', 'Temperature')
linear_regression_with_chart(humidity, apparent_temperature, True, 'Humidity and Alternate Temperature Relationship', 'Humidity', 'Alternate Temperature')


