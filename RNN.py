import pandas as pd
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from config import Configuration 
import numpy as np

config = Configuration() # configuration object

class Data:

    def __init__(self):

        self.df = pd.DataFrame()

    def read(self):
        
        #read data from csv 
        self.df = pd.read_csv('./data/SP500.csv', sep = ';')


class RNN:

    def __init__(self, df):

        self.df = df

        self.X_train = []
        self.X_valid = []
        self.X_test = []

        self.y_train = []
        self.y_valid = []
        self.y_test = []

        self.model = None

        self.test_pred = []
        self.test_pred_flat = []
        self.y_test_flat = []


    def TrainTestSplit(self):

        '''
        Time series split

        Provides train/test indices to split time series data samples that are observed at fixed time intervals, in train/test sets. 
        In each split, test indices must be higher than before, and thus shuffling in cross validator is inappropriate.

        '''
        X = []
        y = []

        tscv = TimeSeriesSplit(gap=0, max_train_size = config.INPUT_SIZE, n_splits=int(len(self.df)/config.OUTPUT_SIZE) - int(config.INPUT_SIZE / config.OUTPUT_SIZE), test_size = config.OUTPUT_SIZE)
        
        for X_index, y_index in tscv.split(self.df[config.PRICE_COL]):

            X.append([self.df[config.PRICE_COL].iloc[i] for i in X_index])
            y.append([self.df[config.PRICE_COL].iloc[i] for i in y_index])

        '''
        Train / Validation / Test set

            With a validation set, you're essentially taking a fraction of your samples out of your training set, 
            or creating an entirely new set all together, and holding out the samples in this set from training.
            During each epoch, the model will be trained on samples in the training set but will NOT be trained on samples in the validation set. 
            Instead, the model will only be validating on each sample in the validation set.

            The purpose of doing this is for you to be able to judge how well your model can generalize. 
            Meaning, how well is your model able to predict on data that it's not seen while being trained.

        '''

        total_sequences = len(X)

        # define train, val, test sizes from config
        train_size = int(total_sequences * config.TRAIN_SIZE)
        val_size = train_size + int(total_sequences * config.VAL_SIZE)
        test_size = val_size + int(total_sequences * config.TEST_SIZE)

        # define train, val, test sets for X and y
        X_train, y_train = np.array(X[:train_size]), np.array(y[:train_size])
        X_valid, y_valid = np.array(X[train_size:val_size]), np.array(y[train_size:val_size])
        X_test, y_test = np.array(X[val_size:test_size]), np.array(y[val_size:test_size])

        # reshape to 3 dimensional -> (batch x timesteps x features) required for RNN input
        self.X_train, self.y_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1])), np.reshape(y_train, (y_train.shape[0], 1, y_train.shape[1]))
        self.X_valid, self.y_valid = np.reshape(X_valid, (X_valid.shape[0], 1, X_valid.shape[1])), np.reshape(y_valid, (y_valid.shape[0], 1, y_valid.shape[1]))
        self.X_test, self.y_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1])), np.reshape(y_test, (y_test.shape[0], 1, y_test.shape[1]))
    

    def trainModel(self):

        '''
            We'll define a sequential model and add the SimpleRNN layer by defining the input shapes. 
            We'll add Dense layers with ReLU activations, set the output layer dimension, and compile the model with Adam optimizer.

            Source: https://www.datatechnotes.com/2020/01/multi-output-multi-step-regression.html

        '''

        self.model = tf.keras.models.Sequential() # initialize sequential model

        self.model.add(tf.keras.layers.SimpleRNN(64, input_shape=(None, config.INPUT_SIZE), activation = 'relu')) # add input layer
        self.model.add(tf.keras.layers.Dense(64, activation="relu")) # add dense layer with X internal units (neurons)
        self.model.add(tf.keras.layers.Dense(config.OUTPUT_SIZE)) # add dense layer as output layer

        opt = tf.keras.optimizers.Adam(learning_rate=0.01) # define optimizer

        self.model.compile(loss="mean_squared_error", optimizer = opt) # compile model

        self.model.summary() # print model summary

        history = self.model.fit(self.X_train, self.y_train, epochs = 100, validation_data = (self.X_valid, self.y_valid)) # train model

        self.model.evaluate(self.X_valid, self.y_valid) # evaluate model

        self.test_pred = self.model.predict(self.X_test) # predict test set


    def performance(self):

        '''
        Mean squared error (matrix comparison)

        - with ax = 0 the average is performed along the row, for each column, returning an array
        - with ax = 1 the average is performed along the column, for each row, returning an array
        - with ax = None the average is performed element-wise along the array, returning a scalar value

        '''
        self.y_test = np.array(self.y_test)[:,0,:] # remove 3th dimension from test set

        ax = None
        rmse = np.sqrt(((self.y_test - self.test_pred)**2).mean(axis=ax))
        mae = abs(self.y_test - self.test_pred).mean(axis=ax)

        print("\nRMSE: {}".format(round(rmse,2)), "\nMAE: {}".format(round(mae,2)))


    def visualize(self):

        '''
        Visualize predictions test set

        '''

        for i in range(len(self.test_pred)):
            self.test_pred_flat.extend(self.test_pred[i])
            self.y_test_flat.extend(self.y_test[i])

        plt.plot([i for i in range(len(self.y_test_flat))], self.y_test_flat, label = 'Actual')
        plt.plot([i for i in range(len(self.test_pred_flat))], self.test_pred_flat, label = 'Predicted')

        # Set the x & y axis labels
        plt.xlabel('Date')
        plt.ylabel('Stock price')

        # Set a title of the current axes.
        plt.title('Stock price prediction using RNN')

        # show a legend on the plot
        plt.legend()
        plt.show()


    def writeToFile(self):
        
        df_result = pd.DataFrame(np.array([self.y_test_flat, self.test_pred_flat]).T.tolist(), columns = ['Actual', 'Prediction'])

        df_result.to_excel('stock_price_pred.xlsx')



if __name__ == "__main__":

    data = Data() # create data object
    data.read() # read stock data

    rnn = RNN(data.df) # create model object

    rnn.TrainTestSplit() # prepare dataset for training
    rnn.trainModel() # train model

    rnn.performance() # print performance 
    rnn.visualize() # visualize prediction output

    rnn.writeToFile() # write predictions to file


    