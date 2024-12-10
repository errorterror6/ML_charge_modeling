
import loader
import parameters
import training
import init

if __name__ == '__main__':
    init.load_data()
    init.init_shjnn()
    training.train()
