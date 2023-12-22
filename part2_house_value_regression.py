import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing, impute, metrics, model_selection

import copy
import sys

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

class Regressor():

    @staticmethod
    def _init_weights(layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0)

    def __init__(self, x, nb_epoch = 1000, neurons = [8, 8, 8], learning_rate = 0.001, loss_fun = "mse", batch_size = 64):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Values stored for pre-processing
        self.x = x
        # Perfoms min-max scaling on x and y values
        self.x_norm = preprocessing.MinMaxScaler() 
        self.y_norm = preprocessing.MinMaxScaler() 
        # Handles the missing values in the data, setting them to a default value
        self.x_imp = impute.SimpleImputer(missing_values=pd.NA, strategy='mean')
        self.label = preprocessing.LabelBinarizer() # Create binary class labels
        self.label.classes_ = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
        self.string_imp = None # Used to handle empty ocean_proximities using one-hot encoding 

        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.neurons = neurons
        self.nb_epoch = nb_epoch
        
        layers = []
        n_in = self.input_size
        for layer in neurons:
            layers.append(nn.Linear(n_in, layer))
            n_in = layer
        layers.append(nn.Linear(n_in, self.output_size))


        # Use ReLU as output layer activation function
        layers.append(nn.ReLU()) 
        
        self.network = nn.Sequential(*layers)
        self.network.apply(self._init_weights)
        self.network.double()

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.prompt_stop = 73
        self.best_epoch = -1
        self.loss_layer = nn.MSELoss()        
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        if training: 
            self.x = x # Need to store x for GridSearchCV
            self.string_imp = x['ocean_proximity'].mode()[0]
        x['ocean_proximity'] = x.loc[:, ['ocean_proximity']].fillna(value=self.string_imp)
        proximity = self.label.transform(x['ocean_proximity'])
        x = x.drop('ocean_proximity', axis=1)
        x = x.join(pd.DataFrame(proximity))

        if training: self.x_imp.fit(x)

        x = self.x_imp.transform(x)
        # If training we initialise our normalisation values
        if training:
            self.x_norm.fit(x)
            if isinstance(y, pd.DataFrame): self.y_norm.fit(y)

        x = torch.from_numpy(self.x_norm.transform(x))
        if isinstance(y, pd.DataFrame):
            y = torch.from_numpy(self.y_norm.transform(y))
        else: 
            y = None
        return x, y


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y, val_x = None, val_y = None):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Sets parameters for early stopping
        min_loss = float("inf")
        best = self

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        X_size = X.size()[0]
        # Returns the with the lowest value into batch_size to avoid oversizing

        batch_size = min(self.batch_size, X_size)
        # Implements Adam algorithm
        optimizer = torch.optim.Adam(self.network.parameters(), lr = self.learning_rate)

        for epoch in range(self.nb_epoch):
            batch_mse_loss = []

            # Use random batches each epoch
            permutation = torch.randperm(X_size)

            for x in range(0, X_size, batch_size):
                # Sets gradients of all model parameters to zero.

                optimizer.zero_grad()

                # Select batch
                indices = permutation[x : x + batch_size]
                batch_X = X[indices]
                batch_Y = Y[indices]

                # Forward and backward pass through particular batch

                output = self.network(batch_X)
                loss = self.loss_layer(output, batch_Y)
                loss.backward()
                optimizer.step()
                
                # Calucates predicted y - tensor separated in order to use numpy 
                y_hat = self.y_norm.inverse_transform(output.detach().numpy())
                y_gold = y.to_numpy()[indices]

                # Calculates mean squared error regression loss for particular batch
                mse = metrics.mean_squared_error(y_gold, y_hat, squared=False)
                batch_mse_loss.append(mse)
                

            # Using validation set for early stopping
            if val_x is not None and val_y is not None:
                val_loss = self.score(val_x, val_y)

                if val_loss < min_loss:
                    self.best_epoch = epoch
                    best = copy.deepcopy(self)
                    min_loss = val_loss
                else:
                    if epoch - self.best_epoch > self.prompt_stop:
                        return best

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        output = self.network(X).detach().numpy()
        inv_trans = self.y_norm.inverse_transform(output)
        return inv_trans


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        output = self.network(X).detach().numpy()

        y_hat = self.y_norm.inverse_transform(output)
        y_gold = y.to_numpy()
        mse = metrics.mean_squared_error(y_gold, y_hat, squared=False)
        return mse

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    #functions for testing
    def get_params(self, deep=True):
        return {
            'x': self.x,
            'learning_rate': self.learning_rate,
            'nb_epoch': self.nb_epoch,
            'neurons': self.neurons,
            'batch_size': self.batch_size
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        
        return self


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(x, y, params): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
        - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    test = 8
    val = 1
    train = 1

    x_size = len(x.index)
    fold_size = x_size // (test + val + train)

    permutation = torch.randperm(x_size)
    val_split = permutation[fold_size * test:fold_size * (test + val)]
    train_split = permutation[fold_size * (test + val):]

    x_train = x.iloc[train_split]

    x_val = x.iloc[val_split]
    y_val = y.iloc[val_split]

    grid = model_selection.GridSearchCV(
        Regressor(x=x_train), 
        param_grid=params, 
        n_jobs=-1, # Set n_jobs to -1 for use all processors
        scoring='neg_root_mean_squared_error',
        verbose=2, 
        return_train_score=True)

    grid.fit(x, y, val_x=x_val, val_y=y_val)

    original_stdout = sys.stdout
    filename = "results.txt"
    res = pd.DataFrame(grid.cv_results_)
    print(f"Saving results to {filename}")

    with open(filename, "w") as outfile:
        sys.stdout = outfile
        print(res[['param_batch_size', 'param_learning_rate', 'param_neurons', 'mean_test_score', 'std_test_score', 'mean_train_score']])
        sys.stdout = original_stdout

    print("Grid scores on val set:")
    print(grid.best_score_)
    print("Best learning rate:", grid.best_estimator_.learning_rate)
    print("Best neuron layout:", grid.best_estimator_.neurons)
    print("Stopping epoch:", grid.best_estimator_.best_epoch)
    
    return  grid.best_params_

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv") 

    # Spliting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    test = 8
    val = 1
    train = 1

    x_size = len(x.index)
    fold_size = x_size // (test + val + train)

    permutation = torch.randperm(x_size)
    test_split = permutation[:fold_size * test]
    val_split = permutation[fold_size * test:fold_size * (test + val)]
    train_split = permutation[fold_size * (test + val):]

    x_train = x.iloc[train_split]
    y_train = y.iloc[train_split]

    x_val = x.iloc[val_split]
    y_val = y.iloc[val_split]

    x_test = x.iloc[test_split]
    y_test = y.iloc[test_split]
        

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 272, neurons = [8,8,8], learning_rate = 0.001, batch_size = 256)
    regressor.fit(x_train, y_train, x_val, y_val)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))

    # Test predict
    result = regressor.predict(x_test)
    print(result)

    # Print stopping time
    print("Stopping time:", regressor.self.best_epoch)


if __name__ == "__main__":
    example_main()

