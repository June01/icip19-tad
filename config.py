""" This is the configuration document used for ICIP paper.
"""

from tools import load_video_length

import argparse

parser = argparse.ArgumentParser()

# Argument to be customized by user
parser.add_argument('--ctx_num', type=int, default=2, help='Context information should be involved in video clips(#ctx_num units before and after video clips)')
parser.add_argument('--unit_size', default=16.0, help='Number of frames in each unit')
parser.add_argument('--unit_feature_size', type=int, default=4096, help='Feature dimension for each unit')

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lambda_reg', type=float, default=1.0)
parser.add_argument('--batch_size', default=128)
parser.add_argument('--max_steps', type=int, default=50003, help='Maximum steps during training')
parser.add_argument('--test_steps', default=50000, help='Model saving steps during training')

parser.add_argument('--action_class_num', default=20, help='Number of actions to be classified')

parser.add_argument('--bsp_level', type=int, default=8, help='Check the paper for more information(three settings: 8(2/4/2), 16(4/8/4), 32(8/16/8))')
parser.add_argument('--dropout', type=bool, choices=[True, False], default=False, help='Flag to use dropout or not')
parser.add_argument('--feat_type', type=str, choices=['BSP', 'Pool', 'SSN'], default='Pool', help='Feature representation methods')
parser.add_argument('--pool_level', type=int, choices=[1,2,3,4,5,6,7,10,14], default=2, help='Temporal pooling granularity')
parser.add_argument('--fusion_type', type=str, choices=['early', 'rgb', 'flow'], default='early', help='Two-stream feature fusion methods')
parser.add_argument('--opm_type', type=str, choices=['adam', 'adam_wd'], default='adam', help='Optimizer')
parser.add_argument('--norm', type=str, choices=['l2', 'No'], default='l2', help='L2 normalization flag')
parser.add_argument('--l1_loss', type=bool, choices=[True, False], default=False, help='Add l1 loss to make the prediction score as small as possible')

parser.add_argument('--issave', type=str, choices=['Yes', 'No'], default='Yes')
parser.add_argument('--ispretrain', type=str, choices=['Yes', 'No'], default='No', help='Training with ')
parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='Flag to indicate stage: train/test')
parser.add_argument('--postfix', type=str, default='test', help='Specify the name of save model')

# Specific for testing
parser.add_argument('--test_model_path', type=str, default='None', help='Path to the model to be tested')
parser.add_argument('--test_iter', type=str, default='50000', help='Specific model test step')

parser.add_argument('--cas_step', type=int, default=1, help='Number of cascade step')
parser.add_argument('--prop_method', choices=['origin', 'bsn'], default='origin', help='The proposals to be used')


args = parser.parse_args()

class Config(object):
    """Class to save all the confugurations
    """
    def __init__(self):

        # Initialize all the arguments
        self.ctx_num = args.ctx_num
        self.unit_size = args.unit_size

        self.lr = args.lr
        self.lambda_reg = args.lambda_reg
        self.batch_size = args.batch_size
        self.max_steps = args.max_steps
        self.test_steps = args.test_steps

        self.action_class_num = args.action_class_num

        self.bsp_level = args.bsp_level
        self.dropout = args.dropout
        self.feat_type = args.feat_type
        self.pool_level = args.pool_level
        self.fusion_type = args.fusion_type
        self.opm_type = args.opm_type
        self.norm = args.norm
        self.l1_loss = args.l1_loss

        self.issave = args.issave
        self.ispretrain = args.ispretrain
        self.mode = args.mode

        self.test_model_path = args.test_model_path
        self.test_iter = args.test_iter

        self.cas_step = args.cas_step
        self.prop_method = args.prop_method

        # The model name to be saved
        self.save_name = 'DET_'+str(args.fusion_type)+'_'+str(args.feat_type)+'_'+str(args.pool_level)+'_ctxnum_'+str(args.ctx_num)+'_lambda_'+str(args.lambda_reg)+'_lr_'+str(args.lr)+'_norm_'+str(args.norm)+'_'+str(args.l1_loss)+'_'+str(args.dropout)+'_'+str(args.opm_type)+'_'+str(args.postfix)


        if args.fusion_type == 'early':
            self.unit_feature_size = args.unit_feature_size
        else:
            self.unit_feature_size = int(args.unit_feature_size/2)

        if args.feat_type == 'Pool':
            self.visual_feature_dim = self.unit_feature_size*(2+(args.pool_level))
        elif args.feat_type == 'SSN':
            self.visual_feature_dim = self.unit_feature_size*7
        else:
            self.visual_feature_dim = self.unit_feature_size*args.bsp_level

        # Path to training samples
        self.train_clip_path = "./val_training_samples.txt"
        self.background_path = "./background_samples.txt"

        # Path to the feature of dataset
        self.prefix = "/data/th14_feature_CBR/"
        self.train_flow_feature_dir = self.prefix+"val_fc6_16_overlap0.5_denseflow/"
        self.train_appr_feature_dir = self.prefix+"val_fc6_16_overlap0.5_resnet/"
        self.test_flow_feature_dir = self.prefix+"test_fc6_16_overlap0.5_denseflow/"
        self.test_appr_feature_dir = self.prefix+"test_fc6_16_overlap0.5_resnet/"

        # Path to the proposals generated from proposal network
        if args.prop_method == 'origin':
            self.test_clip_path = "./props/test_proposals_from_TURN.txt"
        else:
            # self.test_clip_path = './props/thumos14_result_bsn_300.txt'
            self.test_clip_path = "./props/results_TURN_flow_Pool_5_lambda2.0_lr0.005_ctxnum4_test_iter20000_500.txt"

        self.test_len_dict = load_video_length('thumos14_video_length_test.txt')

        self.cat_index_dict={
            0:("Background",0),
            1:("BaseballPitch",7),
            2:("BasketballDunk",9),
            3:("Billiards",12),
            4:("CleanAndJerk",21),
            5:("CliffDiving",22),
            6:("CricketBowling",23),
            7:("CricketShot",24),
            8:("Diving",26),
            9:("FrisbeeCatch",31),
            10:("GolfSwing",33),
            11:("HammerThrow",36),
            12:("HighJump",40),
            13:("JavelinThrow",45),
            14:("LongJump",51),
            15:("PoleVault",68),
            16:("Shotput",79),
            17:("SoccerPenalty",85),
            18:("TennisSwing",92),
            19:("ThrowDiscus",93),
            20:("VolleyballSpiking",97)
        }
