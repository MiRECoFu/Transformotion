import torch
from collections import defaultdict
import torch.optim as optim
# import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from models.transformers.token_embedding import MixEmbedding
from models.transformers.transformotion import Transformotion
from utils.utils import *
from os.path import join as pjoin
from utils.eval_t2m import evaluation_mask_transformer, evaluation_res_transformer
from models.transformers.tools import *

from einops import rearrange, repeat
import wandb

def def_value():
    return 0.0


class TransformotionTrainer:
    def __init__(self, args, transformotion: Transformotion, vq_model):
        self.opt = args
        self.transformotion = transformotion
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()
        # self.embeddings: MixEmbedding = MixEmbedding(vq_model=vq_model, device=self.opt.device)
        wandb.init(
            # set the wandb project where this run will be logged
            project="Transformotion_decoder",

            # track hyperparameters and run metadata
            config={
            "learning_rate": self.opt.lr,
            "epochs": self.opt.max_epoch,
            }
        )

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)
    
    def save(self, file_name, ep, total_it):
        t2m_trans_state_dict = self.transformotion.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            'transformotion': t2m_trans_state_dict,
            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_t2m_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def forward(self, batch_data, mems):

        captions, motions, m_lens = batch_data
        m_lens = m_lens.detach().long().to(self.device)
        # captions = captions.to(self.device)
        motions = motions.detach().float().to(self.device)
        code_idx, _ = self.vq_model.encode(motions)
        # code_idx = code_idx[:, :-1, :]
        # print(code_idx)
        # m_lens = m_lens // 2
        # motion_ids, input_ids, last_hidden_states, labels = self.embeddings.mix_embedding_ids(captions, motions)
        # print("labels============================", labels)

        # (b, n, q)
        # code_idx, _ = self.vq_model.encode(motion)

        # conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        # loss_dict = {}
        # self.pred_ids = []
        # self.acc = []

        ce_loss, acc, pred_id, output, logits, mems = self.transformotion(captions, code_idx[..., 0], m_lens, mems=mems)
        wandb.log({"Train/loss": ce_loss, "Train/acc": acc})

        return ce_loss, acc
    
    def update(self, batch_data, mems):
        loss, acc = self.forward(batch_data, mems)

        self.opt_t2m_transformer.zero_grad()
        loss.backward()
        self.opt_t2m_transformer.step()
        # torch.nn.utils.clip_grad_norm_(self.transformotion.parameters(), 0.25)
        self.scheduler.step()

        return loss.item(), acc

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.transformotion.load_state_dict(checkpoint['transformotion'], strict=False)
        assert len(unexpected_keys) == 0
        # assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']
    
    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        self.transformotion.to(self.device)
        self.vq_model.to(self.device)

        self.opt_t2m_transformer = optim.AdamW(self.transformotion.parameters(), lr=self.opt.lr, weight_decay=1e-6)
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_t2m_transformer,
        #                                                 milestones=self.opt.milestones,
        #                                                 gamma=self.opt.gamma)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt_t2m_transformer,
                                        1000000, eta_min=0)
        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_mask_transformer(
            self.opt.save_root, eval_val_loader, self.transformotion, self.vq_model, self.logger, epoch,
            best_fid=100, best_div=100,
            best_top1=0, best_top2=0, best_top3=0,
            best_matching=100, eval_wrapper=eval_wrapper,
            plot_func=plot_eval, save_ckpt=False, save_anim=True
        )
        best_acc = 0.

        while epoch < self.opt.max_epoch:
            self.transformotion.train()
            self.vq_model.eval()
            mems = tuple()
            for i, batch in enumerate(train_loader):
                it += 1
                # if it < self.opt.warm_up_iter:
                #     self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                
                loss, acc = self.update(batch_data=batch, mems=mems)
                logs['loss'] += loss
                logs['acc'] += acc
                logs['lr'] += self.opt_t2m_transformer.param_groups[0]['lr']
                wandb.log({"Train/lr": self.opt_t2m_transformer.param_groups[0]['lr']})
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    # self.logger.add_scalar('val_loss', val_loss, it)
                    # self.l
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            epoch += 1

            # print('Validation time:')
            # self.vq_model.eval()
            # self.transformotion.eval()

            # val_loss = []
            # val_acc = []
            # with torch.no_grad():
            #     for i, batch_data in enumerate(val_loader):
            #         loss, acc = self.forward(batch_data, mems)
            #         val_loss.append(loss.item())
            #         val_acc.append(acc)

            # print(f"Validation loss:{np.mean(val_loss):.3f}, accuracy:{np.mean(val_acc):.3f}")

            # self.logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
            # self.logger.add_scalar('Val/acc', np.mean(val_acc), epoch)
            # wandb.log({'Val/loss': np.mean(val_loss), 'Val/acc': np.mean(val_acc)})

            # if np.mean(val_acc) > best_acc:
            #     print(f"Improved accuracy from {best_acc:.02f} to {np.mean(val_acc)}!!!")
            #     self.save(pjoin(self.opt.model_dir, 'net_best_acc.tar'), epoch, it)
            #     best_acc = np.mean(val_acc)
            if epoch % 10 == 0:
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_mask_transformer(
                    self.opt.save_root, eval_val_loader, self.transformotion, self.vq_model, self.logger, epoch, best_fid=best_fid,
                    best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                    best_matching=best_matching, eval_wrapper=eval_wrapper,
                    plot_func=plot_eval, save_ckpt=True, save_anim=True
                )