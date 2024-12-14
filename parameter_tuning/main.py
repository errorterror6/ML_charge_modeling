
import loader
import parameters
import training
import init

import sys
import time

if __name__ == '__main__':
    
    #command line options
    
    if not ((len(sys.argv) == 2) or (len(sys.argv) == 3)) :
        print("usage: python main.py <name> <desc=none>")
        print("args given: ", sys.argv)
        exit(1)
        
    if (sys.argv[1] == 'clear'):
        loader.clear_saves()
        print("Logs: Main: Cleared all saves.")
        exit(0)
    
    #setting up saving path
        
    parameters.model_params['name'] = sys.argv[1]
    if(len(sys.argv) == 3): parameters.model_params['desc'] = sys.argv[2]
    else: parameters.model_params['desc'] = f'epochs_{parameters.model_params["total_epochs_train"]}_beta_{parameters.model_params["beta"]}_ld_{parameters.model_params["latent_dim"]}_batch_{parameters.model_params["n_batch"]}'
    
    #begin training loop
    
    time_start = time.time()
    
    init.load_data()
    init.init_shjnn(parameters.model_params)
    training.train(parameters.model_params, parameters.dataset)
    
    time_end = time.time()
    
    #end training loop
    
    #report training time.
    time_taken = time.strftime('%H:%M:%S', time.gmtime(time_end - time_start))
    time_per_epoch = time.strftime('%H:%M:%S', time.gmtime((time_end - time_start)/parameters.model_params['total_epochs_train']))
    print(f"Total training time: {time_taken}")
    print(f"Average time per epoch: {time_per_epoch}")
