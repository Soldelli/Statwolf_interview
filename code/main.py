# import libraries
import numpy as np
import pandas as pd
from os import getcwd
from sys import exit, exc_info
import random as rnd
import matplotlib
import datetime
from sklearn import preprocessing
from sklearn.externals import joblib
import matplotlib.pyplot as plt


# Directory path
code_path   = getcwd()
data_path   = code_path[0:code_path.rfind("code")]+'data'
output_path = code_path[0:code_path.rfind("code")]+'output'

#Functions
def load_data(filename):
    ''' The function perform the data loading from the file specified by the input variable filename which must be put
    into the data directory

    Exceptuion during data loading are handled

    Outputs: two numpy array, extracted from the file'''

    try:
        df = pd.read_csv(filepath_or_buffer=data_path+'\\'+ filename, sep=',')
        dates = df['DATE'].values
        income = df['y'].values
    except:
        print('Some error occurred during data loading.')
        exit(0)

    print('Data successfully loaded')
    # check for missing values in data
    missing_income_values, missing_dates_values = np.sum(np.isnan(income)), np.sum(isinstance(dates, datetime.date))
    if missing_income_values:
        print('Number of missing income values ' + str(missing_income_values) + ' out of ' + str(len(income)) + ' entries')
    if missing_dates_values:
        print('Number of missing income values '+ str(missing_dates_values)+ ' out of ' +str(len(dates))+ ' entries')

    #conversion to datetimes
    for i in range(len(dates)):
        dates[i] = datetime.date(year= int(dates[i][0:4]), month = int(dates[i][5:7]), day= int(dates[i][8:10]))
    # determine delta between days
    delta = []
    for i in range(len(dates)-1):
        delta.append(dates[i+1]- dates[i])
        if delta[-1].days > 1:
            print(dates[i],dates[i+1],)
    #sum of deltas
    temp=0
    for i in range(len(delta)): temp+=delta[i].days
    print('Number of missing days ',temp-len(delta))

    return dates, income


def pre_processing(income_new,nan_rm_tech):
    ''' The function perform data preprocessing, addressing missing values and normalization problems.

    Missing values are added through different techniques, depending on the nan_rm_tech parameters which
    sele specifiesd technique.
    0 - Fill the gaps with the mean value of the feature
    1 - Fill the gaps with the median of the feature, more resilient to outliers
    2 - Linear interpolation, or average of neighbor values.
    3 - Windowed linear interpolation, or windowed average of neighbor values

    Features are rescaled in the range [0-1]

    Outputs: preprocessed features and list of nan indicies (for later use in data_exploration)'''

    # address missing value problem
    nan_idx = np.where(np.isnan(income_new))[0] # nan idx determination

    if nan_rm_tech ==0:
        income_new[nan_idx] = np.nanmean(income_new)
    if nan_rm_tech ==1:
        income_new[nan_idx] = np.nanmedian(income_new)
    if nan_rm_tech == 2:
        for i in nan_idx:
            if i > 0 and i < len(income_new):
                if ~np.isnan(income_new[i + 1]):
                    income_new[i] = (income_new[i - 1] + income_new[i + 1]) / 2
                else:
                    ii=i+2
                    while np.isnan(income_new[ii]): ii += 1
                    income_new[i] = (income_new[i - 1] + income_new[ii]) / 2
            else:
                if i == 0: income_new[i] = income_new[i + 1]
                else:      income_new[i] = income_new[i - 1]
    if nan_rm_tech == 3:
        print('To be done')

    # data normalization, scale data between 0 and 1
    income_new = income_new.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    income_new = min_max_scaler.fit_transform(income_new)

    # scaler save to file
    joblib.dump(min_max_scaler, output_path+'\\models\\scaler.pkl')
    return income_new, nan_idx


def model_training(x, prediction_lenght, select_model):
    train_idx = len(x)//2
    valid_idx = len(x)//4 + train_idx
    x_train = x[0:train_idx]
    x_validation = x[train_idx:valid_idx]

    print(len(x),len(x_train),len(x_validation))

    # regression model
    # if model == 0:
    #     #neural network parameters
    #     num_neurons_per_layer = [10,20,10,prediction_lenght]
    #     activation_functions = ['relu']
    #     bias = 1
    #     weights = 1
    #
    #
    #
    # if model == 1:
    #     print('To be done. Probabilmente XGBoost')



def test():
    a=0
    return a


def data_exploration(dates, income_new, nan_idx,nan_rm_tech, save_figure):
    '''Plot of income variable, missing values (which have been previously added) are enlightened '''
    plt.figure()
    plt.xlabel('Days'), plt.ylabel('Income'), plt.title('Trend visualization')
    plt.grid(True), plt.hold

    x = np.linspace(0,len(dates)+1, len(dates))
    plt.plot(x,income_new,linewidth=0.5, marker='o', markersize=1)

    plt.plot(x[nan_idx], income_new[nan_idx],  color='red', marker='o', markersize=2, linestyle='')
    if save_figure:
        plt.savefig(output_path+'\\images\\data_exploration_'+str(nan_rm_tech)+'.pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()


#Main
if __name__ == '__main__':
    # Parameters
    filename = 'ts_forecast.csv'
    save_figure = False
    nan_rm_tech = 2     # technique for nan removal, 0 mean of all vlaue, 1- meadi of all value, 2- mean of pre
                        # and post vale 3- windowed mean
    prediction_lenght = 30
    select_model = 0     # 0 neural network, 1 other model

    # Functions call
    dates, income = load_data(filename)
    income_new, nan_idx = pre_processing(income,nan_rm_tech)

    data_exploration(dates,income_new,nan_idx,nan_rm_tech, save_figure)
    model_training(income_new, prediction_lenght, select_model)

    # to load: scaler = joblib.load(output+'\\models\\scaler.pkl)'