
''' imports '''

# orchestration functions
from .orch import *

# import custom dataset
from .data import *

# import model components
from .model import *

# import training components
from .train import *

# import inference components
from .infer import *

# dimensionality reduction
#from .analysis import *



''' TODO

- update data ingestion format, separate environment from state variables (exclude env for training)

- prepare visualisation of changing inference (trajectory prediction) during training checkpoints

- update latent space variation maps (trajectory as function of latent space variables)

- optimise dynamics model network hyperparameters, compare training and inference performance of a set

- impliment symbolic regression on learned dynamics model



'''
