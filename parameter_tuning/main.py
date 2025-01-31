print("Loading....")
import loader
import parameters
import training
import init

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
    parameters.trainer = input("Enter the trainer to use (B-VAE, RNN): ")
    print("Logs: Main: Using trainer: ", parameters.trainer)


    #begin training loop
    
    time_start = time.time()
    
    init.load_data()
    init.init_shjnn(parameters.model_params)
    init.init_RNN(parameters.model_params)
    training.train(parameters.model_params, parameters.dataset)


    
    time_end = time.time()
    
    #end training loop
    
    #report training time.
    time_taken = time.strftime('%H:%M:%S', time.gmtime(time_end - time_start))
    time_per_epoch = time.strftime('%H:%M:%S', time.gmtime((time_end - time_start)/parameters.model_params['total_epochs_train']))
    print(f"Total training time: {time_taken}")
    print(f"Average time per epoch: {time_per_epoch}")
