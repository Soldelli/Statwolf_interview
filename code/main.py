# import libraries
import numpy as np
import pandas as pd
from os import getcwd, sep, system, environ
from sys import exit, exc_info, platform
import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import time


# neural network
import keras
from keras.models       import Model, Sequential
from keras.layers       import LSTM, Dropout, GRU, Reshape, Input, Dense, Flatten, Reshape
from keras.optimizers   import Nadam, SGD
from keras.initializers import Constant, VarianceScaling
from keras.callbacks    import TensorBoard
from keras              import backend as K

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # disable any tensorflow warning

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
    print('\n---- Data loading phase ----')
    try:
        print(data_path + sep + filename)
        df = pd.read_csv(filepath_or_buffer=data_path + sep + filename, sep=',')  # read data file
        #dates = pd.to_datetime(df['DATE'])  # separate the two features
        dates  = df['DATE'].values           # separate the two features
        income = df['y'].values
    except:
        print('Some error occurred during data loading.')
        exit(0)


    print('Data successfully loaded')

    return dates, income


def pre_processing(dates, income, nan_rm_tech):
    ''' The function perform data preprocessing, addressing missing values and normalization problems.

    Missing date values are added manually to obtain one day spacing through the all dates.

    Missing income values are added through different techniques, depending on the nan_rm_tech parameters which
    sele specifiesd technique.
    0 - Fill the gaps with the mean value of the feature
    1 - Fill the gaps with the median of the feature, more resilient to outliers
    2 - Linear interpolation, or average of neighbor values.
    3 - Windowed linear interpolation, or windowed average of neighbor values

    Features are rescaled in the range [0-1]

    Outputs: preprocessed features and list of nan indicies (for later use in data_exploration)'''

    print('\n---- Data preprocessing phase ----')
    #### address missing value problem  -------------------------------------

    # conversion of first feature (dates) to datetimes and fill missing dates
    for i in range(len(dates)):
        dates[i] = datetime.datetime.strptime(dates[i], "%Y-%m-%d").date()

    # dates filling procedure
    dates_temp, income_temp, missing_days = [], [], []  # temp variables for dates filling procedure
    for i in range(len(dates) - 1):
        dates_temp.append(dates[i])             # append real value
        income_temp.append(income[i])           # append real value
        if (dates[i + 1] - dates[i]).days > 1:  # previously verified that only single days are missing,
            # no consequenly missing dates are present in the dataset
            missing_days.append(dates[i] + datetime.timedelta(days=1))  # save missing days for later displays
            dates_temp.append(missing_days[-1]) # append missing date
            income_temp.append(np.nan)          # append nan value related to missing date
    dates_temp.append(dates[-1])                # append last value
    income_temp.append(income[-1])

    dates, income = np.asarray(dates_temp), np.asarray(income_temp)  # overwrite variables

    # check for missing values of income
    missing_income_values, missing_dates_values = np.sum(np.isnan(income)), len(missing_days)
    perc1, perc2 = (missing_income_values / len(income)) * 100, (len(missing_days) / len(dates_temp)) * 100
    if missing_income_values:
        print('Number of missing income values: ' + str(missing_income_values) + ' out of ' +
              str(len(income)) + ' entries, (' + "{0:.2f}".format(perc1) + '%)')
    if missing_dates_values:
        print('Number of missing date values: ' + str(missing_dates_values) + ' out of ' +
              str(len(dates)) + ' entries, (' + "{0:.2f}".format(perc2) + '%)')

    nan_idx = np.where(np.isnan(income))[0] # nan idx determination

    if nan_rm_tech ==0:
        income[nan_idx] = np.nanmean(income)
    if nan_rm_tech ==1:
        income[nan_idx] = np.nanmedian(income)
    if nan_rm_tech == 2:
        for i in nan_idx:
            if i > 0 and i < len(income):
                if ~np.isnan(income[i + 1]):
                    income[i] = (income[i - 1] + income[i + 1]) / 2
                else:
                    ii=i+2
                    while np.isnan(income[ii]): ii += 1
                    income[i] = (income[i - 1] + income[ii]) / 2
            else:
                if i == 0: income[i] = income[i + 1]
                else:      income[i] = income[i - 1]
    if nan_rm_tech == 3:
        print('To be done')

    ### data normalization, scale data between 0 and 1 ----------------------
    income = income.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    income = min_max_scaler.fit_transform(income)

    # scaler saved to file --------------------------------------------------

    joblib.dump(min_max_scaler, output_path +sep+'models'+sep+'scaler.pkl')

    return dates, income, nan_idx


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning."""

	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def model_training(x, prediction_lenght, select_model):
    # model parameters -------------------------------------------------------------------------------------------------
    num_features = 1
    input_window_size  = 10  # number of samples per channel used to feed the NN
    output_window_size = 10
    batch_size         = 25
    training_epochs    = 10

    # Training set construction ----------------------------------------------------------------------------------------
    # The following function takes advantage of dataframe shift function to create sliding windowd representation of the
    # time series
    print('Mean value of data sequence:', np.mean(x))
    dataset = np.asarray(series_to_supervised(x, n_in=input_window_size,
                                              n_out=output_window_size, dropnan=True).values.tolist())
    # The previously obtained dataset is divided into training and testing examples
    idx = int(len(dataset)*0.8)
    X_test, y_test =  dataset[idx:,:input_window_size], dataset[idx:,input_window_size:]    # last 20% of dataset is reserved for testing
    # the other 80% is divided again in 80% training 20% validation
    X_train, X_val, y_train, y_val = train_test_split( dataset[:idx,:input_window_size], dataset[:idx,input_window_size:],
                                                        test_size = 0.20)

    X_train = np.reshape(X_train,(len(X_train),input_window_size,1))
    X_val   = np.reshape(X_val,  (len(X_val),  input_window_size, 1))
    X_test  = np.reshape(X_test, (len(X_test), input_window_size, 1))

    print('\n----  Training phase ----')
    print(X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape)

    # Neural Network parameters ----------------------------------------------------------------------------------------
    RNN_neurons = [50, 50]  # Defines the number of neurons of the recurrent layers
    full_conn   = [input_window_size, output_window_size]  # Number of neurons of dense layers (fully connected)
    dropout     = [0.05,0.05]  # Definition of Dropout probabilities
    activation  = ['relu', 'tanh']

    # Definition of initializers for the weigths and biases of the dense layers.
    kernel_init = VarianceScaling(scale=1.0, mode='fan_avg', distribution='normal', seed=None)
    bias_init = Constant(value=0.1)


    # Neural Network model ---------------------------------------------------------------------------------------------
    model = Sequential()

    #model.add(Reshape((input_window_size,num_features), input_shape=([input_window_size])))

    # Layer 1
    model.add(GRU(units            = RNN_neurons[0],
                  activation       = activation[1],  # tanh
                  kernel_initializer=kernel_init,
                  use_bias         = True,
                  bias_initializer = bias_init,
                  dropout          = dropout[0],
                  return_sequences = True,  # set to true is following layer is recurrent
                  input_shape      = (None, input_window_size, num_features),
                  batch_input_shape= [None, input_window_size, num_features],
                  batch_size       = None,
                  stateful         = False))

    # Layer 2
    model.add(GRU(units            = RNN_neurons[1],
                  activation       = activation[1],
                  kernel_initializer=kernel_init,
                  use_bias         = True,
                  bias_initializer = bias_init,
                  dropout          = dropout[1],
                  return_sequences = False,  # set to true is following layer is recurrent
                  stateful         = False))

    #model.add(Flatten())


    # Layer 3
    model.add(Dense(units      = full_conn[1],
                    activation = 'linear',
                    use_bias   = True,
                    kernel_initializer = kernel_init,
                    bias_initializer   = bias_init))

    #model.summary()                     # print details of the neural network

    # Optimizer setup --------------------------------------------------------------------------------------------------
    #opt = Nadam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    opt = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)


    model.compile(optimizer = opt,
                  loss      = 'mse',    # Loss function specification - MSE
                  metrics   = ['mape'])  # additional metric of analysis
    # purposes, but still usefull in debugging


    # Callback definition for tensorboard usage ------------------------------------------------------------------------
    tbCallback = TensorBoard(log_dir=output_path+sep+'models'+sep, histogram_freq=0,
                             batch_size=batch_size, write_graph=True,
                             write_grads=True, write_images=False,
                             embeddings_freq=0, embeddings_layer_names=None,
                             embeddings_metadata=None)


    # Training ---------------------------------------------------------------------------------------------------------
    # If stateless RNN are used, standard fit is employed.
    history = model.fit(x               =X_train,  # X data
                        y               =y_train,  # Y data
                        epochs          =training_epochs,      # number of fit iteration across all training set
                        batch_size      =batch_size,           # number of training samples preprocessed in parallel.
                        verbose         =2,                    # 0 for no logging, 1 for progress bar logging, 2 for one log line per epoch.
                        #validation_split=0.2,                  # float (0. < x < 1). Fraction of the data to use as held-out validation data.
                        validation_data = (X_val,y_val),
                        shuffle         =False,                # data is not shuffled from epoch to epoch.
                        callbacks       =[tbCallback])         # save graph and other data for visualization with tensorboard.


    # Model saving -----------------------------------------------------------------------------------------------------
    #model.save(output_path +sep+ 'models'+sep+'RNN_model.pkl')  # save model within the directory specified by path

    # Performances evaluation ------------------------------------------------------------------------------------------
    score = model.evaluate(X_test, y_test, verbose=2, batch_size=batch_size)
    print('Test loss loss: ' + "{0:.5f}".format(score[0]))

    prediction_visualization(X_test,y_test,model)

    # Tensorboard invocation -------------------------------------------------------------------------------------------
    #system('tensorboard --logdir=' + output_path +sep+ 'models --host=127.0.0.1')


def data_exploration(dates_new, income_new, nan_idx,nan_rm_tech, save_figure):
    '''Plot of income variable, missing values (which have been previously added) are enlightened '''
    plt.figure()
    plt.xlabel('Days'), plt.ylabel('Income'), plt.title('Trend visualization')
    plt.grid(True), plt.hold


    x = np.linspace(0,len(dates_new)+1, len(dates_new))
    plt.plot(x,income_new,linewidth=0.5, marker='o', markersize=1)

    plt.plot(x[nan_idx], income_new[nan_idx],  color='red', marker='o', markersize=2, linestyle='')
    plt.legend(['true values', 'fake values'], loc='upper right')
    if save_figure:
        plt.savefig(output_path +sep+'images'+sep+'data_exploration_' + str(nan_rm_tech) + '.pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def prediction_visualization(X,y,model):
    for i in range(10):
        y_hat = model.predict(np.reshape(X[i],(1,X[i].shape[0],X[i].shape[1])))   # fare in blocco, e gestire dopo il plot
        plt.figure()
        plt.xlabel('Days'), plt.ylabel('Income'), plt.title('Trend visualization')
        plt.grid(True), plt.hold

        x = np.linspace(0, len(y[i]) + 1, len(y[i]))
        plt.plot(x, y[i], linewidth=0.5, marker='o', markersize=1)

        plt.plot(x, y_hat[0], linewidth=0.5, color='red', marker='o', markersize=1)
        plt.legend(['true serie', 'predicted serie'], loc='upper right')

        plt.savefig(output_path + sep + 'images' + sep + 'data_prediction_' + str(i) + '.pdf',bbox_inches='tight')
        plt.close()



#Main
if __name__ == '__main__':
    # Parameters --------------------------------
    filename = 'ts_forecast.csv'
    save_figure = True
    nan_rm_tech = 2     # technique for nan removal, 0 mean of all vlaue, 1- meadi of all value, 2- mean of pre
                        # and post vale 3- windowed mean
    prediction_lenght = 30
    select_model = 0     # 0 neural network, 1 other model

    # Functions call ----------------------------
    t=time.time()
    dates, income = load_data(filename)
    print('Performed in ' + "{0:.2f}".format(time.time()-t) + ' seconds.')
    t = time.time()
    dates_new, income_new, nan_idx = pre_processing(dates,income,nan_rm_tech)
    print('Performed in ' + "{0:.2f}".format(time.time() - t)+ ' seconds.')
    #t = time.time()
    #data_exploration(dates_new,income_new,nan_idx,nan_rm_tech, save_figure)
    #print(time.time() - t)
    t = time.time()
    model_training(income_new, prediction_lenght, select_model)
    print('Performed in ' + "{0:.2f}".format(time.time() - t)+ ' seconds.')
    t = time.time()

    # to load: scaler = joblib.load(output+sep+'models'+sep+'scaler.pkl)'