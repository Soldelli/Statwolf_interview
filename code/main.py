# import libraries
import numpy  as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import time
from os         import getcwd, sep, system, environ, path, makedirs
from sys        import exit, exc_info, platform
from sklearn    import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.externals       import joblib
from scipy      import signal
from dateutil   import relativedelta

# neural network
import keras
import keras.objectives
from keras.models       import Model, Sequential, load_model
from keras.layers       import LSTM, GRU, Reshape, Input, Dense, Flatten, Reshape, Activation, Dropout, Input, Concatenate
from keras.layers.normalization       import BatchNormalization
from keras.optimizers   import Nadam, SGD, Adam
from keras.initializers import Constant, VarianceScaling
from keras.callbacks    import TensorBoard, Callback
from keras.utils        import plot_model
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


def data_exploration(dates_new, income_new, nan_idx,nan_rm_tech, montly_view,save_figure):
    '''Plot of income variable, missing values (which have been previously added) are enlightened '''
    print('\n----  Data exploration and visualization ----')
    if montly_view:
        print('Plot of single months sequence')
    month_idx = []
    year_idx = [0]
    # montly plot
    if not path.exists(output_path + sep + 'images' + sep + 'montly'):  # create directory is not present
        makedirs(output_path + sep + 'images' + sep+ 'montly')
    date = []
    date2 = []

    # formula below computes the number of month between first and last data as: delta_years*12 + delta_monts
    for i in range(dates_new[-1].month-dates_new[0].month+(dates_new[-1].year-dates_new[0].year)*12):
        if i == 0:
            date  = dates_new[0]
            date2 = dates_new[0].replace(month=date.month + 1)
        else:
            try:
                date = date.replace(month=date.month + 1)
            except ValueError:
                if date.month == 12:
                    date = date.replace(year=date.year + 1, month=1)
                    year_idx.append(np.argwhere(dates_new == date)[0][0])
            try:
                date2= date.replace(month=date.month + 1)
            except ValueError:
                if date2.month == 12:
                    date2 = date2.replace(year=date2.year + 1, month=1)

        idx1 = np.argwhere(dates_new == date)[0][0]
        month_idx.append(idx1)
        idx2 = np.argwhere(dates_new == date2)[0][0]

        if montly_view:
            plt.figure(),
            plt.xlabel('Days ('+str(date.year)+'/'+str(date.month)+')' ), plt.ylabel('Income'), plt.title('Trend visualization')
            plt.grid(True), plt.hold

            x = np.linspace(0, len(dates_new[idx1:idx2]) + 1, len(dates_new[idx1:idx2]))
            plt.plot(x, income_new[idx1:idx2], linewidth=0.5, marker='o', markersize=1)

            plt.savefig(output_path + sep + 'images' + sep + 'montly'+ sep +'data_exploration_montly_view' + str(i) + '.pdf',
                            bbox_inches='tight',dpi=600)
            plt.close()

    #whole sequence
    print('Plot of whole sequence')
    plt.figure(figsize=(15,6))
    plt.xlabel('Days'), plt.ylabel('Income'), plt.title('Trend visualization')
    plt.grid(True), plt.hold
    plt.xticks(year_idx, dates_new[year_idx],  fontsize=8)
    plt.yticks(fontsize=8)
    x = np.linspace(0, len(dates_new) + 1, len(dates_new))
    plt.plot(x, income_new, linewidth=0.5, marker='o', markersize=0.6)
    plt.plot(x[nan_idx], income_new[nan_idx], color='red', marker='o', markersize=2, linestyle='')
    for idx in month_idx:
        plt.axvline(idx, color='green', linestyle=':', linewidth=0.25)
    plt.axvline(len(x), color='green', linestyle=':', linewidth=0.25)
    for idx in year_idx:
        plt.axvline(idx, color='purple', linestyle='--', linewidth=1.5)
    plt.legend(['true values', 'fake values'], loc='upper right')
    if save_figure:
        plt.savefig(output_path + sep + 'images' + sep + 'data_exploration_' + str(nan_rm_tech) + '.pdf',
                    bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.show()


def pre_processing(dates, income, nan_rm_tech,filter):
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

    print('Missing data filled with technique ' +str(nan_rm_tech))

    income_new = income
    if filter :
        print('Filtering procedure perdormed with low pass Butterworth filter')

        # PSD estimation - do we have white noise?
        nyq = 0.5 * 1                       # Filtering parameters
        normalCutoff = 0.04 / nyq
        b, a = signal.butter(4, normalCutoff, btype='low', analog=False)
        income_new = signal.filtfilt(b, a, income)

        x1, p1 = signal.welch(x=income, window="hanning", axis=0, detrend=False,
                              nperseg=len(income_new) / 4)      # utilization of welch method for Power Spectral
                                                                # Density estimation
        x2, p2 = signal.welch(x=income_new, window="hanning", axis=0, detrend=False,
                                nperseg=len(income_new) / 4)

        p1,p2 = 20 * np.log10(p1), 20 * np.log10(p2)
        plt.semilogx(x1, p1, linewidth=1)
        plt.semilogx(x2, p2, linewidth=1, color='red')
        plt.ylabel('Power Spectral Density [dB]')
        plt.xlabel('Frequency [Hz]')
        plt.grid(True, which='both')
        plt.ylim([np.min(p1)-10, np.max(p1)+10])
        plt.savefig(output_path + sep + 'images' + sep + 'PSD_time_series.pdf', dpi=150, transparent=False)
        plt.close()

    ### data normalization, scale data between 0 and 1 ----------------------
    income_new = income_new.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    income_new = min_max_scaler.fit_transform(income_new)

    # scaler saved to file --------------------------------------------------

    joblib.dump(min_max_scaler, output_path +sep+'models'+sep+'scaler.pkl')

    return dates, income_new, nan_idx


def series_to_supervised(data, n_in, n_out, dropnan=True):
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


def dataset_split(x, w_size, split, ensemble, num_model):
    '''The function splits the dataset into training, validation and test'''
    # Training set construction ----------------------------------------------------------------------------------------
    # The following function takes advantage of dataframe shift function to create sliding windowd representation of the
    # time series
    datase = []
    if num_model == 1:
        print('\n----  Performing data split  ----')
        dataset = np.asarray(series_to_supervised(x, n_in=w_size[0], n_out=w_size[1], dropnan=True).values.tolist())
    if num_model == 2:
        print()
        dataset = np.asarray(series_to_supervised(x, n_in=w_size[0]+365, n_out=w_size[1], dropnan=True).values.tolist())
    elif num_model ==3:
        print()
        dataset = np.asarray(series_to_supervised(x, n_in=365, n_out=w_size[1], dropnan=True).values.tolist())

    print('Model number', num_model, 'win_in', w_size[0], 'w_out', w_size[1])
    # The previously obtained dataset is divided into training and testing examples
    idx = int(len(dataset) * split[0])
    X_test, y_test = dataset[idx:, :w_size[0]], dataset[idx:,-w_size[1]:]  # last 20% of dataset is reserved for testing

    # the other 80% is divided again in 80% training 20% validation
    X_train, X_val, y_train, y_val = train_test_split(dataset[:idx, :w_size[0]],
                                                      dataset[:idx, -w_size[1]:],test_size=split[1])

    X_train = np.reshape(X_train, (len(X_train), w_size[0], 1))
    X_val   = np.reshape(X_val,   (len(X_val),   w_size[0], 1))
    X_test  = np.reshape(X_test,  (len(X_test),  w_size[0], 1))

    print('Training data shape   X=', X_train.shape,'y=', y_train.shape,)
    print('Validation data shape X=', X_val.shape,  ' y=', y_val.shape)
    print('Test data shape       X=', X_test.shape, ' y=', y_test.shape)

    return X_train, X_test, X_val, y_train, y_test, y_val, idx


def dataset_split2(x, w_size, split, ensemble):
    '''The function splits the dataset into training, validation and test'''
    # Training set construction ----------------------------------------------------------------------------------------
    # The following function takes advantage of dataframe shift function to create sliding windowd representation of the
    # time series
    datase = []

    print('\n----  Performing data split  ----')
    dataset = np.asarray(series_to_supervised(x, n_in=w_size[0], n_out=w_size[1], dropnan=True).values.tolist())

    print('Window sizes: win_in', w_size[0], 'w_out', w_size[1])

    # The previously obtained dataset is divided into training and testing examples
    idx = int(len(dataset) * split[0])
    X_test, y_test = dataset[idx:, :w_size[0]], dataset[idx:,-w_size[1]:]  # last 20% of dataset is reserved for testing

    # the other 80% is divided again in 80% training 20% validation
    X_train, X_val, y_train, y_val = train_test_split(dataset[:idx, :w_size[0]],
                                                      dataset[:idx, -w_size[1]:],test_size=split[1])

    X_train_opt1, X_val_opt1, y_train_opt1, y_val_opt1 = X_train[:,:12],  X_val[:,:12],  y_train[:,:12],  y_val[:,:12]
    X_train_opt2, X_val_opt2, y_train_opt2, y_val_opt2 = X_train[:,9:21], X_val[:,9:21], y_train[:,9:21], y_val[:,9:21]
    X_train_opt3, X_val_opt3, y_train_opt3, y_val_opt3 = X_train[:,18:],  X_val[:,18:],  y_train[:,18:],  y_val[:,18:]

    X_train_opt1 = np.reshape(X_train_opt1, (len(X_train_opt1), 12, 1))
    X_val_opt1   = np.reshape(X_val_opt1,   (len(X_val_opt1),   12, 1))
    X_train_opt2 = np.reshape(X_train_opt2, (len(X_train_opt2), 12, 1))
    X_val_opt2   = np.reshape(X_val_opt2,   (len(X_val_opt2),   12, 1))
    X_train_opt3 = np.reshape(X_train_opt3, (len(X_train_opt3), 12, 1))
    X_val_opt3   = np.reshape(X_val_opt3,   (len(X_val_opt3),   12, 1))


    X_train = np.reshape(X_train, (len(X_train), w_size[0], 1))
    X_val   = np.reshape(X_val,   (len(X_val),   w_size[0], 1))
    X_test  = np.reshape(X_test,  (len(X_test),  w_size[0], 1))

    print('Training data shape   X=', X_train.shape,'y=', y_train.shape,)
    print('Validation data shape X=', X_val.shape,  ' y=', y_val.shape)
    print('Test data shape       X=', X_test.shape, ' y=', y_test.shape)

    opt=[X_train_opt1, X_val_opt1, y_train_opt1, y_val_opt1,
         X_train_opt2, X_val_opt2, y_train_opt2, y_val_opt2,
         X_train_opt3, X_val_opt3, y_train_opt3, y_val_opt3]

    return X_train, X_test, X_val, y_train, y_test, y_val, idx, opt


def MAPE(Y, Yhat):
    diff = K.abs((Y - Yhat) / K.clip(K.abs(Y),
                                            K.epsilon(),
                                            None))
    return 100. * K.mean(diff, axis=-1)


def SMAPE(Y, Yhat):
    # symmetric mean absolute percentage error, they call it symmetric (but it is not)
    # https://robjhyndman.com/hyndsight/smape/
    divide = K.abs(Y - Yhat) / K.abs(Y + Yhat)
    smape = 100. *  (K.mean(2. * divide))
    return smape


def model_training(X_train, X_val, y_train, y_val, w_size, num_model, arch_type, opt_data):
    '''
    The function trains a recursive neural network with 2 GRU layes and a fully connected one, to predict a future
    windows of the signal.
    :param X_train:
    :param X_val:
    :param y_train:
    :param y_val:
    :param w_size:
    '''
    if num_model == 1:
        print('\n----  Training phase ----')
    if arch_type == 0:
        print('Training model '+str(num_model)+ '\n')
    else:
        print('Training model Multi Branch Neural Network\n')
    # model parameters -------------------------------------------------------------------------------------------------
    num_features = 1
    input_window_size  = w_size[0]  # number of samples per channel used to feed the NN
    output_window_size = w_size[1]
    batch_size         = 4
    training_epochs    = 1

    # Neural Network parameters ----------------------------------------------------------------------------------------
    RNN_neurons = [50, 50]  # Defines the number of neurons of the recurrent layers
    full_conn   = [input_window_size, output_window_size,12,50,50]  # Number of neurons of dense layers (fully connected)
    dropout     = [0.25,0.5]  # Definition of Dropout probabilities
    activation  = ['relu','tanh', 'sigmoid','linear']

    # Definition of initializers for the weigths and biases of the dense layers.
    kernel_init = VarianceScaling(scale=1.0, mode='fan_avg', distribution='normal', seed=None)
    bias_init = Constant(value=0.1)

    model, opt, loss, metrics = [], [], [], []
    X_train_opt1, X_val_opt1, y_train_opt1, y_val_opt1 = [],[],[],[]
    X_train_opt2, X_val_opt2, y_train_opt2, y_val_opt2 = [],[],[],[]
    X_train_opt3, X_val_opt3, y_train_opt3, y_val_opt3 = [],[],[],[]

    try:
        X_train_opt1, X_val_opt1, y_train_opt1, y_val_opt1,\
               X_train_opt2, X_val_opt2, y_train_opt2, y_val_opt2,\
                    X_train_opt3, X_val_opt3, y_train_opt3, y_val_opt3 = opt_data        # unpacking optional data
    except:
        if arch_type==1:
            print('Optional data unpacking problem.')
            exit(0)

    if arch_type == 0:
        # Neural Network model ---------------------------------------------------------------------------------------------
        model = Sequential()

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

        # Layer 3
        model.add(Dense(units      = full_conn[1],
                        activation = activation[2],
                        use_bias   = True,
                        kernel_initializer = kernel_init,
                        bias_initializer   = bias_init))

        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        # opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

        # Losses
        loss = ['mean_squared_error', 'mean_absolute_error',
                'mean_squared_logarithmic_error']

        # Metrics
        metrics = ['mse', SMAPE]  # and loss
        keras.objectives.custom_loss = SMAPE

    elif arch_type == 1:

        input = Input(shape=(input_window_size, num_features))

        #Branch 1   ------
        #x = GRU(units=RNN_neurons[0], kernel_initializer=kernel_init, use_bias=False,return_sequences=True)(input)
        x = GRU(units=RNN_neurons[0], kernel_initializer=kernel_init, use_bias=False)(input)
        x = Activation(activation[0])(x)
        #x = BatchNormalization()(x)
        #x = Dropout(dropout[0])(x)
        #x = GRU(units=RNN_neurons[0], kernel_initializer=kernel_init, use_bias=False)(x)
        #x = Activation(activation[0])(x)
        #x = BatchNormalization()(x)
        x = Dropout(dropout[0])(x)
        x = Dense(units=full_conn[3], kernel_initializer=kernel_init, use_bias=False)(x)
        x = Activation(activation[0])(x)
        #x = BatchNormalization()(x)
        x = Dropout(dropout[0])(x)
        x1 = Dense(units=full_conn[2], kernel_initializer=kernel_init, use_bias=False)(x)
        Branch1 = Activation(activation[2], name='Branch1')(x1)

        # Branch 2   ------
        y = GRU(units=RNN_neurons[0], kernel_initializer=kernel_init, use_bias=False,return_sequences=True)(input)
        y = Activation(activation[0])(y)
        #y = BatchNormalization()(y)
        y = Dropout(dropout[0])(y)
        y = GRU(units=RNN_neurons[0], kernel_initializer=kernel_init, use_bias=False)(y)
        y = Activation(activation[0])(y)
        #y = BatchNormalization()(y)
        y = Dropout(dropout[0])(y)
        y = Dense(units=full_conn[3], kernel_initializer=kernel_init, use_bias=False)(y)
        y = Activation(activation[0])(y)
        #y = BatchNormalization()(y)
        y = Dropout(dropout[0])(y)
        y1 = Dense(units=full_conn[2], kernel_initializer=kernel_init, use_bias=False)(y)
        Branch2 = Activation(activation[2], name='Branch2')(y1)

        # Branch 3   ------
        z = GRU(units=RNN_neurons[0], kernel_initializer=kernel_init, use_bias=False,return_sequences=True)(input)
        z = Activation(activation[0])(z)
        #z = BatchNormalization()(z)
        z = Dropout(dropout[0])(z)
        z = GRU(units=RNN_neurons[0], kernel_initializer=kernel_init, use_bias=False,return_sequences=True)(z)
        z = Activation(activation[0])(z)
        z = Dropout(dropout[0])(z)
        z = GRU(units=RNN_neurons[0], kernel_initializer=kernel_init, use_bias=False)(z)
        z = Activation(activation[0])(z)
        #z = BatchNormalization()(z)
        z = Dropout(dropout[0])(z)
        z = Dense(units=full_conn[3], kernel_initializer=kernel_init, use_bias=False)(z)
        z = Activation(activation[0])(z)
        #z = BatchNormalization()(z)
        z = Dropout(dropout[0])(z)
        z1 = Dense(units=full_conn[2], kernel_initializer=kernel_init, use_bias=False)(z)
        Branch3 = Activation(activation[2], name='Branch3')(z1)

        # Merge branches
        merge = Concatenate(axis=-1)([x,y,z]) #([Branch1,Branch2,Branch3])
        w = Dense(units=full_conn[3], kernel_initializer=kernel_init, use_bias=False)(merge)
        w = Activation(activation[1])(w)
        w = Dropout(dropout[0])(w)
        w = BatchNormalization()(w)
        w = Dense(units=full_conn[1], kernel_initializer=kernel_init, use_bias=False)(w)
        output = Activation(activation[2], name='output')(w)


        model = Model(inputs=input, outputs=[Branch1,Branch2,Branch3,output], name='MultiBranchNetwork')
        #model.summary()


        #opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

        # Losses
        # ['mean_squared_error', 'mean_absolute_error','mean_squared_logarithmic_error']
        loss = {'Branch1' : 'mean_squared_error',
                'Branch2' : 'mean_squared_error',
                'Branch3' : 'mean_squared_error',
                'output'  : 'mean_squared_error'}

        lossWeights = { 'Branch1' : .3,
                        'Branch2' : .6,
                        'Branch3' : 1.0,
                        'output'  : .3}

        # Metrics
        metrics = [SMAPE]  # and loss
        keras.objectives.custom_loss = SMAPE


    else:
        print('No valid model has been selected')

    #model.summary()                     # print details of the neural network

    # Optimizer setup --------------------------------------------------------------------------------------------------

    model.compile(optimizer     = opt,
                  loss          = loss,
                  loss_weights  = lossWeights,
                  metrics       = metrics)  # additional metric of analysis


    # purposes, but still usefull in debugging


    # Callback definition for tensorboard usage ------------------------------------------------------------------------
    tbCallback = TensorBoard(log_dir=output_path+sep+'models'+sep, histogram_freq=0,
                             batch_size=batch_size, write_graph=True,
                             write_grads=True, write_images=False,
                             embeddings_freq=0, embeddings_layer_names=None,
                             embeddings_metadata=None)


    # Training ---------------------------------------------------------------------------------------------------------
    # If stateless RNN are used, standard fit is employed.
    history = []
    if arch_type == 0:
        history = model.fit(x               =X_train,  # X data
                            y               =y_train,  # Y data
                            epochs          =training_epochs,        # number of fit iteration across all training set
                            batch_size      =batch_size,             # number of training samples preprocessed in parallel.
                            verbose         =2,                      # 0 for no logging, 1 for progress bar logging, 2 for one log line per epoch.
                            #validation_split=0.2,                   # float (0. < x < 1). Fraction of the data to use as held-out validation data.
                            validation_data = (X_val,y_val),
                            shuffle         =False,                  # data is not shuffled from epoch to epoch.
                            callbacks       =[tbCallback]) # save graph and other data for visualization with tensorboard.
    elif arch_type == 1:
        history = model.fit(x               =X_train,  # X data
                            y               ={'Branch1':y_train_opt1, 'Branch2':y_train_opt2,
                                              'Branch3':y_train_opt3, 'output':y_train },  # Y data
                            epochs          =training_epochs,       # number of fit iteration across all training set
                            batch_size      =batch_size,            # number of training samples preprocessed in parallel.
                            verbose         =2,                     # 0 for no logging, 1 for progress bar logging, 2 for one log line per epoch.
                            validation_data =(X_val, {'Branch1':y_val_opt1, 'Branch2':y_val_opt2,
                                             'Branch3':y_val_opt3, 'output':y_val }),
                            shuffle         =False,  # data is not shuffled from epoch to epoch.
                            callbacks       =[tbCallback])  # save graph and other data for visualization with tensorboard.

    score = model.evaluate(x=X_train, y={'Branch1':y_train_opt1, 'Branch2':y_train_opt2,
                                         'Branch3':y_train_opt3, 'output':y_train }, verbose=0, batch_size=batch_size)
    # print('\nTraining loss: mae= '+"{0:.2e}".format(score[0])+ ', and metrics mse= '+"{0:.2e}".format(score[1])+
    #       ' SMAPE='+"{0:.3f}".format(score[2]))
    print('\nTraining loss: mse= ' + "{0:.2e}".format(score[0]) + ', and metrics SMAPE=' + "{0:.3f}".format(score[-1]))

    score = model.evaluate(x=X_val, y={'Branch1':y_val_opt1, 'Branch2':y_val_opt2,
                                       'Branch3':y_val_opt3, 'output':y_val }, verbose=0, batch_size=batch_size)
    # print('Validation loss: mae= '+"{0:.2e}".format( score[0])+ ', and metrics mse= '+"{0:.2e}".format(score[1])+
    #       ' SMAPE= '+"{0:.2f}".format(score[2]))

    print('Validation loss: mse= ' + "{0:.2e}".format(score[0]) + ', and metrics SMAPE= ' + "{0:.2f}".format(score[-1]))

    # Model saving -----------------------------------------------------------------------------------------------------
    model.save(output_path +sep+ 'models'+sep+'RNN_model_'+str(num_model)+'.pkl')  # save model within the directory specified by path
    plot_model(model, to_file=output_path +sep+ 'images'+sep+'Network_model_plot.pdf')

    # Tensorboard invocation -------------------------------------------------------------------------------------------
    #system('tensorboard --logdir=' + output_path +sep+ 'models --host=127.0.0.1')
    print('\ntensorboard --logdir=' + output_path +sep+ 'models --host=127.0.0.1\n')


def model_test(X,y,num_model, arch_type, opt_data):
    '''
    The function perform the test score assessment, and plots the true signal vs the predicted one, for each model trained.
    :param X:
    :param y:
    :return:
    '''
    if num_model == 1:
        print('\n----  Test phase  ----')
    print('Testing model ' + str(num_model))
    # Performances evaluation ------------------------------------------------------------------------------------------
    model = load_model(output_path + sep + 'models' + sep + 'RNN_model_'+str(num_model)+'.pkl', custom_objects={'SMAPE': SMAPE})
    #score = model.evaluate(X, y, verbose=0, batch_size=2)
    #print('Test loss: mae= '+"{0:.2e}".format(score[0])+ ', and metrics mse= '+"{0:.2e}".format(score[1])+ ' SMAPE= '+
    #      "{0:.2f}".format( score[2]) + '\n')

    # Predictions visualization ----------------------------------------------------------------------------------------
    if not path.exists(output_path + sep + 'images' + sep + 'predictions'):  # create directory is not present
        makedirs(output_path + sep + 'images' + sep+ 'predictions')

    print(X.shape,X[0::y.shape[1]].shape,y.shape, y[0::y.shape[1]].shape, model.predict(X[0::y.shape[1]])[-1].shape)

    y_new = np.reshape(y[0::y.shape[1]],-1)
    y_hat = []
    if not arch_type:
        y_hat = np.reshape(model.predict(X[0::y.shape[1]]), -1)
    else:
        y_hat = np.reshape(model.predict(X[0::y.shape[1]])[-1],-1)

    plt.figure()
    plt.xlabel('Days'), plt.ylabel('Income'), plt.title('Prediction comparison')
    plt.grid(True), plt.hold

    x = np.linspace(0, len(y_new) + 1, len(y_new))
    plt.plot(x, y_new, linewidth=0.5, marker='o', markersize=1)

    plt.plot(x, y_hat, linewidth=0.5, color='red', marker='o', markersize=1)
    plt.legend(['true serie', 'predicted serie'], loc='upper right')
    plt.ylim([0, 1])
    plt.savefig(output_path + sep + 'images' + sep + 'data_prediction_'+str(num_model)+'.pdf', bbox_inches='tight', dpi=600)
    plt.close()


def recursive_prediction(X,idx,w_size, w_size2, w_size3, ensemble):
    '''
    The function takes the previously trained model and perform the ensembling while execute recursive forecasting/predictions
    using predicted values as new input (just for model 1). Other two models refer to past data which is available.

    GRAND SCHEME EXPLANATION: All models predicts next 10 days, then a new input data for model 1 is create by
    using the 10 predicted values and last 20 samples of previous serie. For other two models we just shift the windows
    and it is not necessary to use predicted values in next step prediction.
    This gives strenght and resiliance to model 1 problems due to derive in predictions.

    :param X1:
    :param X2:
    :param X3:
    :param y1:
    :param y2:
    :param y3:
    :return:
    '''
    print('\n----  Ensemble model investigation  ----')
    model2,model3 = [],[]
    model1 = load_model(output_path + sep + 'models' + sep + 'RNN_model_1.pkl', custom_objects={'SMAPE': SMAPE})
    if ensemble:
        model2 = load_model(output_path + sep + 'models' + sep + 'RNN_model_2.pkl', custom_objects={'SMAPE': SMAPE})
        model3 = load_model(output_path + sep + 'models' + sep + 'RNN_model_3.pkl', custom_objects={'SMAPE': SMAPE})


    y_new =np.reshape(X[idx+w_size[0]:],-1)
    iter =  (X.shape[0]-w_size[1]-idx)//w_size[0]
    print('Number of input iterations: ', iter)
    offset = 365
    input, input2, input3 = [], [], []
    for i in range(iter):
        #print(idx+i*w_size[0],idx+ (i+1)*w_size[0])
        input.append(X[idx+i*w_size[0]: idx+ (i+1)*w_size[0]])

    for i in range(iter*3):
        #print(idx - offset + i * w_size[1], idx - offset + w_size[0] + i * w_size[1])
        input2.append(X[idx - offset + i * w_size[1] :idx - offset + w_size[0] +i * w_size[1]])

    for i in range(iter*3):
        #print(idx - offset + w_size[0] + i * w_size[1], idx - offset + w_size[0] + (i + 1) * w_size[1])
        input3.append(X[idx - offset + w_size[0] + i * w_size[1] : idx - offset + w_size[0] + (i + 1) * w_size[1]])
    # prendere1 input da 30 per ogni mese, dopo di che iterare la predizione con i valori predetti allo step prima

    y_hat=0
    total_sequence=[]
    for i in range(len(input)):
        print(input3[i * 3])
        y_pred1  = model1.predict(np.reshape(input[i],      (1, w_size[0],  1)))
        y2_pred1 = model2.predict(np.reshape(input2[i * 3], (1, w_size2[0], 1)))
        y3_pred1 = model3.predict(np.reshape(input3[i * 3], (1, w_size3[0], 1)))
        avg_pred = (y_pred1[0]+ y2_pred1[0]+ y3_pred1[0])/3

        temp     = np.append(input[i][w_size[1]:], avg_pred)

        y_pred2  = model1.predict(np.reshape(temp,(1, len(temp), 1)))
        y2_pred2 = model2.predict(np.reshape(input2[i * 3 + 1], (1, w_size2[0], 1)))
        y3_pred2 = model3.predict(np.reshape(input3[i * 3 + 1], (1, w_size3[0], 1)))
        avg_pred = (y_pred2[0] + y2_pred2[0] + y3_pred2[0]) / 3

        temp     = np.append(temp[w_size[1]:], avg_pred)

        y_pred3  = model1.predict(np.reshape(temp,(1, len(temp), 1)))
        y2_pred3 = model2.predict(np.reshape(input2[i * 3 + 2], (1, w_size2[0], 1)))
        y3_pred3 = model3.predict(np.reshape(input3[i * 3 + 2], (1, w_size3[0], 1)))
        avg_pred = (y_pred3[0] + y2_pred3[0] + y3_pred3[0]) / 3

        y_hat    = np.append(temp[w_size[1]:], avg_pred)
        total_sequence.append(y_hat)

    y_hat = np.reshape(np.asarray(total_sequence),-1)

    nyq = 0.5 * 1  # Filtering parameters
    normalCutoff = 0.06 / nyq
    b, a = signal.butter(4, normalCutoff, btype='low', analog=False)
    y_hat = signal.filtfilt(b, a, y_hat)

    # Density estimation

    plt.figure()
    plt.xlabel('Days'), plt.ylabel('Income'), plt.title('Prediction comparison')
    plt.grid(True), plt.hold

    x = np.linspace(0, y_hat.shape[0] + 1, y_hat.shape[0])
    plt.plot(x, y_new[0:y_hat.shape[0]], linewidth=0.5, marker='o', markersize=1)

    plt.plot(x, y_hat, linewidth=0.5, color='red', marker='o', markersize=1)
    plt.legend(['true serie', 'predicted serie'], loc='upper right')
    plt.ylim([0, 1])
    plt.savefig(output_path + sep + 'images' + sep + 'data_recursive_prediction.pdf', bbox_inches='tight', dpi=600)
    plt.close()


#Main
if __name__ == '__main__':
    # Parameters --------------------------------
    filename = 'ts_forecast.csv'
    save_figure = True
    montly_view = False
    filter      = True
    ensemble    = True
    train       = True
    nan_rm_tech = 2     # technique for nan removal, 0 mean of all vlaue, 1- meadi of all value, 2- mean of pre
                        # and post vale 3- windowed mean
    arch_type = 1

    w_size  = [30, 10]
    w_size2 = [30, 10]
    w_size3 = [10, 10]
    w_size_branch = [30, 30]
    split   = [0.8,0.2] # used to split data, first the dataset is divided in training + validation 80% and test 20%
                        # then validation is chosen as 20% of the first part

    # Functions call ----------------------------
    t=time.time()
    dates, income = load_data(filename)
    print('Media dati'+  "{0:.2f}".format(np.nanmean(income)))
    print('Varianza dati'+  "{0:.2f}".format(np.nanstd(income)))

    print('Performed in ' + "{0:.2f}".format(time.time()-t) + ' seconds.')
    t = time.time()

    dates_new, income_new, nan_idx = pre_processing(dates,income,nan_rm_tech,filter)

    print('Performed in ' + "{0:.2f}".format(time.time() - t)+ ' seconds.')
    t = time.time()

    data_exploration(dates_new, income_new, nan_idx, nan_rm_tech,montly_view, save_figure)

    print('Performed in ' + "{0:.2f}".format(time.time() - t) + ' seconds.')
    t = time.time()

    X_train, X_test, X_val, y_train, y_test, y_val, X_train2, X_test2, X_val2, y_train2, y_test2, y_val2, X_train3, \
    X_test3, X_val3, y_train3, y_test3, y_val3, opt, idx = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], \
                                                      [], [],[], [], []


    if not arch_type:
        X_train, X_test, X_val, y_train, y_test, y_val, idx = dataset_split(income_new,w_size,split,ensemble, num_model=1)

        if ensemble: # if ensemble is enalble then we compute the dataset split also for the additional models
            X_train2, X_test2, X_val2, y_train2, y_test2, y_val2, _ = dataset_split(income_new, w_size2, split, ensemble, num_model=2)
            X_train3, X_test3, X_val3, y_train3, y_test3, y_val3, _ = dataset_split(income_new, w_size3, split, ensemble, num_model=3)
    else:
        X_train, X_test, X_val, y_train, y_test, y_val, idx, opt = \
            dataset_split2(income_new, w_size_branch, split, ensemble)


    print('Performed in ' + "{0:.2f}".format(time.time() - t) + ' seconds.')
    t = time.time()

    if train:
        if not arch_type:
            model_training(X_train, X_val, y_train, y_val, w_size, num_model=1, arch_type=arch_type, opt_data=opt)
            if ensemble:    # if ensemble is enalble then we train also the two additional models
               model_training(X_train2, X_val2, y_train2, y_val2, w_size2, num_model=2, arch_type=arch_type, opt_data=[])
               model_training(X_train3, X_val3, y_train3, y_val3, w_size3, num_model=3, arch_type=arch_type, opt_data=[])
        else:
            model_training(X_train, X_val, y_train, y_val, w_size_branch, num_model=1, arch_type=arch_type, opt_data=opt)

    print('Performed in ' + "{0:.2f}".format(time.time() - t)+ ' seconds.')
    t = time.time()

    if not arch_type:
        model_test(X_test, y_test, num_model=1, arch_type=arch_type, opt_data=opt)
        if ensemble:
            model_test(X_test2, y_test2, num_model=2, arch_type=arch_type, opt_data=opt)
            model_test(X_test3, y_test3, num_model=3, arch_type=arch_type, opt_data=opt)
    else:
        model_test(X_test, y_test, num_model=1, arch_type=arch_type, opt_data=opt)

    print('Performed in ' + "{0:.2f}".format(time.time() - t) + ' seconds.')
    t = time.time()

    if not arch_type:
        recursive_prediction( np.reshape(income_new,-1), idx, w_size, w_size2, w_size3, ensemble)
        print('Performed in ' + "{0:.2f}".format(time.time() - t) + ' seconds.')

    # to load: scaler = joblib.load(output+sep+'models'+sep+'scaler.pkl)'