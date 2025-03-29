print("Loading....")
import sys
sys.path.append('/mnt/c/vscode/thesis/ML_charge_modeling')

import parameter_tuning.loader as loader
import parameter_tuning.parameters as parameters
import parameter_tuning.training as training
import parameter_tuning.init as init
import parameter_tuning.data_dropout as data_dropout

import sys
import time

if __name__ == '__main__':
    
    

    parameters.model_params['name'] = input("Enter a name: ")
    if (parameters.model_params['name'] == 'clear'):
        loader.clear_saves()
        print("Logs: Main: Cleared all saves.")
        exit(0)
    if len(parameters.model_params['name']) == 0:
        parameters.model_params['name'] = 'default'
    parameters.model_params['desc'] = input("Enter a description: ")
    drop_data = input("run with missing data? (y/n): ")
    parameters.trainer = input("Enter the trainer to use (B-VAE, RNN, LSTM): ")
    
    print("Logs: Main: Using trainer: ", parameters.trainer)
    


    #begin training loop
    
    
    
    if drop_data == 'y':
        drop_data = True
    else:
        drop_data = False
    init.load_data(drop_data=drop_data)
    
    # data_dropout.verify_missing_data()
    # exit(0)
    
    init.init_shjnn(parameters.model_params)
    match parameters.trainer:
        case 'B-VAE':       
            parameters.model = init.init_B_VAE(parameters.model_params)
        case 'RNN':
            parameters.model = init.init_RNN(parameters.model_params)
        case 'LSTM':
            parameters.model = init.init_LSTM(parameters.model_params)
        case _:
            print("Logs: Main: Invalid trainer. Exiting.")
            exit(1)
            
    time_start = time.time()    
        
    training.train(parameters.model_params, parameters.dataset)


    
    time_end = time.time()
    
    #end training loop
    
    #report training time.
    time_taken = time.strftime('%H:%M:%S', time.gmtime(time_end - time_start))
    time_per_epoch = time.strftime('%H:%M:%S', time.gmtime((time_end - time_start)/parameters.model_params['total_epochs_train']))
    print(f"Total training time: {time_taken}")
    print(f"Average time per epoch: {time_per_epoch}")
