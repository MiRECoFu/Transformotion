# from options.base_opt import BaseOptions
import argparse

from options.base_opt import BaseOptions

class TrainT2MOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        self.parser.add_argument('--max_epoch', type=int, default=3000, help='Maximum number of epoch for training')
        # self.parser.add_argument('--max_iters', type=int, default=150_000, help='Training iterations')

        '''LR scheduler'''
        self.parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate schedule factor')
        self.parser.add_argument('--milestones', default=[50_000], nargs="+", type=int,
                            help="learning rate schedule (iterations)")
        self.parser.add_argument('--warm_up_iter', default=3000, type=int, help='number of total iterations for warmup')

        '''Condition'''
        self.parser.add_argument('--cond_drop_prob', type=float, default=0.1, help='Drop ratio of condition, for classifier-free guidance')
        self.parser.add_argument("--seed", default=3407, type=int, help="Seed")

        self.parser.add_argument('--is_continue', action="store_true", help='Is this trial continuing previous state?')
        self.parser.add_argument('--gumbel_sample', action="store_true", help='Strategy for token sampling, True: Gumbel sampling, False: Categorical sampling')
        self.parser.add_argument('--share_weight', action="store_true", help='Whether to share weight for projection/embedding, for residual transformer.')

        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress, (iteration)')
        # self.parser.add_argument('--save_every_e', type=int, default=100, help='Frequency of printing training progress')
        self.parser.add_argument('--eval_every_e', type=int, default=10, help='Frequency of animating eval results, (epoch)')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of saving checkpoint, (iteration)')
        self.parser.add_argument("--code_dim", type=int, default=512, help="embedding dimension")
        self.parser.add_argument("--nb_code", type=int, default=512, help="nb of embedding")


        self.is_train = True
        
# python trainers/train_transformotion.py --dataset_name t2m --name trm_xl_b64_d1024_726_clip_nh16_nl18_downt_1 --vq_name rvq_down_t_1_d1024 --gpu_id 0