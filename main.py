import os
from src.interface import Interface

if __name__ == '__main__':
    print('hello world')
    import configparser
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.cfg'))
    
    interface = Interface(config)
    # interface.run_trial_training()
    interface.run_pretrained_training()