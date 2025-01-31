7
''' imports '''

# pytorch components for model
import torch
import torch.nn as nn
import torch.nn.functional as F



''' initialise complete model '''

def init_model(latent_dim, nhidden, rnn_nhidden, obs_dim, nbatch, lr, device = None):

    ''' Function Title

        function details

    Args:
        var (int): some variable

    Returns:
        (str): some output
    '''

    # initialise models (latent ode function, recognition rnn, decoder mlp)
    func = LatentODEfunc(latent_dim, nhidden)
    rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, nbatch)
    dec = Decoder(latent_dim, obs_dim, nhidden)

    # check for cuda device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('intialising model on device: {}'.format(device))

    # push models to device
    func.to(device)
    rec.to(device)
    dec.to(device)


    # aggregate model parameters
    params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))

    # initialise optimiser
    optim = torch.optim.Adam(params, lr = lr)


    # return each model instance
    return func, rec, dec, optim, device



''' model components '''

class RecognitionRNN(nn.Module):

    ''' recognition rnn model

        RNN for trajectory to latent state-space transformation
        ingest trajectory (sequence-sampled state-space) backwards in time
        output time-independent latent variables in latent-space
    '''

    def __init__(self, latent_dim: int = 4, obs_dim: int = 2, nhidden: int = 25, nbatch: int = 1):
        super(RecognitionRNN, self).__init__()

        # set hidden layer and batch
        self.nhidden = nhidden
        self.nbatch = nbatch

        # input and hidden state to new hidden state
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)

        # new hidden state to output, mean and variance of latent vars
        self.h2o = nn.Linear(nhidden, latent_dim * 2)


    def forward(self, x, h):

        # concat input and hidden layers
        combined = torch.cat((x, h), dim = 1)

        # compute new hidden state, apply tanh activation
        h = torch.tanh( self.i2h(combined) )

        # compute output result from new hidden state
        out = self.h2o(h)

        # return result, hidden state
        return out, h


    def initHidden(self):

        # initialise and return hidden state with zeros
        return torch.zeros(self.nbatch, self.nhidden)



class LatentODEfunc(nn.Module):

    ''' latent ode model

        parameterise dynamics function with one-hidden-layer network
    '''

    def __init__(self, latent_dim = 4, nhidden = 20):
        super(LatentODEfunc, self).__init__()

        # non-linear activation
        self.elu = nn.ELU(inplace=True)

        # linear layers for dynamics over latent space
        self.fci = nn.Linear(latent_dim, nhidden)

        # 3 hidden linear layers
        self.fc1 = nn.Linear(nhidden, nhidden)
        #self.fc2 = nn.Linear(nhidden, nhidden)
        #self.fc3 = nn.Linear(nhidden, nhidden)

        self.fco = nn.Linear(nhidden, latent_dim)

        # number function evaluations
        self.nfe = 0


    def forward(self, t, x):

        # update n func eval
        self.nfe += 1

        # forward pass through network, activations
        out = self.fci(x)
        out = self.elu(out)

        out = self.fc1(out)
        out = self.elu(out)

        #out = self.fc2(out)
        #out = self.elu(out)

        #out = self.fc3(out)
        #out = self.elu(out)

        out = self.fco(out)

        return out



class Decoder(nn.Module):

    ''' decoder model

        transform latent-space trajectory to output state-space trajectory
    '''

    def __init__(self, latent_dim = 4, obs_dim = 2, nhidden = 20):
        super(Decoder, self).__init__()

        # non-linear activation
        self.relu = nn.ReLU(inplace=True)

        # linear layers from latent space to inital dimensions
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)


    def forward(self, z):

        # forward pass through network, activations
        out = self.fc1(z)
        out = self.relu(out)

        out = self.fc2(out)

        return out




###
### TODO - update model to CDE
###

''' Neural Controlled Differential Equation (CDE) Model '''


class CDEFunc(torch.nn.Module):

    def __init__(self, input_channels, hidden_channels):

        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################

        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)


    def forward(self, z):
        z = self.linear1(z)
        z = torch.tanh(z)
        z = self.linear2(z)

        ######################
        # The one thing you need to be careful about is the shape of the output tensor. Ignoring the batch dimensions,
        # it must be a matrix, because we need it to represent a linear map from R^input_channels to
        # R^hidden_channels.
        ######################

        z = z.view(*z.shape[:-1], self.hidden_channels, self.input_channels)

        return z


######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################

class NeuralCDE(torch.nn.Module):

    #def __init__(self, input_channels, hidden_channels, output_channels):
    def __init__(self, input_channels, hidden_channels):
        super(NeuralCDE, self).__init__()

        self.hidden_channels = hidden_channels
        self.func = CDEFunc(input_channels, hidden_channels)
        #self.linear = torch.nn.Linear(hidden_channels, output_channels)
        self.linear = torch.nn.Linear(hidden_channels, input_channels)


    def forward(self, times, coeffs):

        ######################
        # Extract the sizes of the batch dimensions from the coefficients
        ######################

        coeff, _, _, _ = coeffs
        batch_dims = coeff.shape[:-2]
        z0 = torch.zeros(*batch_dims, self.hidden_channels, dtype=times.dtype, device=times.device)

        ######################
        # Actually solve the CDE.
        ######################

        z_T = controldiffeq.cdeint(dX_dt=controldiffeq.NaturalCubicSpline(times, coeffs).derivative,
                                   z0=z0,
                                   func=self.func,
                                   t=times[[0, -1]],
                                   atol=1e-2,
                                   rtol=1e-2)

        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value,
        # and then apply a linear map.
        ######################

        z_T = z_T[1]
        pred_y = self.linear(z_T)

        return pred_y

