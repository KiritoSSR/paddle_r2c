import argparse

parser = argparse.ArgumentParser(description='train')
parser.add_argument('-params',dest='params',help='Params location',type=str)
parser.add_argument('-rationale',action="store_true",help='use rationale')
parser.add_argument('-folder',dest='folder',help='folder location',type=str, default='model/saves/flagship_answer')
parser.add_argument('-no_tqdm',dest='no_tqdm',action='store_true')
parser.add_argument('-num_epoch',dest='num_epoch',type=int,default=20)
parser.add_argument('-batch_size',dest='batch_size',help='batch_size',type=int,default=32)
parser.add_argument('-lr',dest='lr',help='learning rate',type=float,default=0.000005)
parser.add_argument('-weight_decay',dest='weight_decay',help='weight_decay',type=float,default=0.0001)
parser.add_argument('-num_serialized_models_to_keep',dest='num_serialized_models_to_keep',help='Params location',type=int,default=2)
parser.add_argument('-grad_norm',dest='grad_norm',help='grad_norm',type=float,default=1.0)
parser.add_argument('-patience',dest='patience',help='patience',type=int,default=6)
parser.add_argument('-scheduler_type',dest='scheduler_type',help='scheduler_type',type=str,default='reduce_on_plateau')
parser.add_argument('-scheduler_factor',dest='scheduler_factor',help='scheduler_factor',type=float,default=0.5)
parser.add_argument('-scheduler_model',dest='scheduler_model',help='scheduler_model',type=str,default='max')
parser.add_argument('-scheduler_patience',dest='scheduler_patience',help='scheduler_patience',type=int,default=1)
parser.add_argument('-scheduler_verbose',dest='scheduler_verbose',help='scheduler_verbose',type=bool,default=True)
parser.add_argument('-scheduler_cooldown',dest='scheduler_cooldown',help='scheduler_cooldown',type=int,default=1)


parser.add_argument('-input_dropout',dest='input_dropout',help='input_dropout',type=float,default=0.3)
parser.add_argument('-hidden_dim_maxpool',dest='hidden_dim_maxpool',help='hidden_dim_maxpool',type=int,default=1024)
parser.add_argument('-pool_question',dest='pool_question',help='pool_question',type=bool,default=True)
parser.add_argument('-pool_answer',dest='pool_answer',help='pool_answer',type=bool,default=True)
parser.add_argument('-args_reset_every',dest='args_reset_every',help='ARGS_RESET_EVERY',type=int,default=300)


args = parser.parse_args()
