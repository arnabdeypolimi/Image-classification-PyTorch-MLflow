from modules.config.config_manager import ConfigurationManager
import sys, getopt
from modules.train.train import Train
from modules.evaluation.matrix import Matrix
from modules.train.helper import Helper
import torch

def main(argv):
   config_file = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["config="])
   except getopt.GetoptError:
      print('main.py --config <name>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('main.py -config <name>')
         sys.exit()
      elif opt in ("-config", "--config"):
         config_file = arg
   print('config file name: '+config_file)

   config = ConfigurationManager.from_file(config_file)

   dataloaders_dict = Helper.dataloader(Helper.input_size(config.model_name), config.data_dir, batch_size=config.batch_size)

   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

   model_ft=Train.run_training(config, dataloaders_dict,device)

   Matrix.create_matrix(config,dataloaders_dict,device,model_ft)

if __name__ == "__main__":
   main(sys.argv[1:])