

import os
import numpy as np
from os.path import join as pjoin
from data_process.motion_dataset import Text2MotionDataset
from models.transformers.token_embedding import MixEmbedding
from models.transformers.transformotion import Transformotion
from utils.fixseed import fixseed
from torch.utils.data import DataLoader

import torch

from models.vq.model import RVQVAE
from options.train_opt import TrainT2MOptions
from utils.get_opt import get_opt
from utils.paramUtil import t2m_kinematic_chain, kit_kinematic_chain
from visualization.joints2bvh import Joint2BVHConvertor
from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric
from einops import rearrange, repeat

def load_vq_model(opt):
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)
    
    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {opt.vq_name}')
    return vq_model, vq_opt


def test_dataset(opt, vq_model):
    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'std.npy'))
    opt.text_dir = pjoin(opt.data_root, 'texts')
    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')
    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file)
    val_dataset = Text2MotionDataset(opt, mean, std, val_split_file)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)
    for i, (captions, motions, lengths) in enumerate(train_loader):
        print(f"Batch {i+1}")
        print("Captions:", captions)

        print("Motions:", motions)
        print("Lengths:", lengths)
        print("="*50)
        for i, (caption) in enumerate(captions):
            mix_embedding(caption, motion=motions[i], vq_model=vq_model)


# if __name__ == "__main__":
    # parser = TrainT2MOptions()
    # opt = parser.parse()
    # fixseed(opt.seed)

    # opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    # torch.autograd.set_detect_anomaly(True)
    # opt.checkpoints_dir = '/root/autodl-tmp/checkpoints'
    # opt.dataset_name = 't2m'
    # # opt.save_root
    # opt.gpu_id = 0


    # opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    # opt.model_dir = pjoin(opt.save_root, 'model')
    # # opt.meta_dir = pjoin(opt.save_root, 'meta')
    # opt.eval_dir = pjoin(opt.save_root, 'animation')
    # opt.log_dir = pjoin('./log/t2m/', opt.dataset_name, opt.name)

    # if opt.dataset_name == 't2m':
    #     opt.data_root = '/root/autodl-tmp/HumanML3D/'
    #     opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    #     opt.joints_num = 22
    #     opt.max_motion_len = 55
    #     dim_pose = 263
    #     radius = 4
    #     fps = 20
    #     kinematic_chain = t2m_kinematic_chain
    #     dataset_opt_path = '/root/autodl-tmp/checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    # os.makedirs(opt.model_dir, exist_ok=True)
    # # os.makedirs(opt.meta_dir, exist_ok=True)
    # os.makedirs(opt.eval_dir, exist_ok=True)
    # os.makedirs(opt.log_dir, exist_ok=True)
    # vq_model, vq_opt = load_vq_model(opt)
    # test_dataset(opt, vq_model)

if __name__ == "__main__":
    parser = TrainT2MOptions()
    opt = parser.parse()
    opt.code_dim = 768
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    opt.num_tokens = opt.nb_code
    fixseed(opt.seed)
    if opt.dataset_name == 't2m':
        opt.data_root = '/root/autodl-tmp/HumanML3D/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 22
        opt.max_motion_len = 55
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = t2m_kinematic_chain
        dataset_opt_path = '/root/autodl-tmp/checkpoints/t2m/transformotion_rvq/opt.txt'
    vq_model, vq_opt = load_vq_model(opt)
    embeddings: MixEmbedding = MixEmbedding(vq_model=vq_model, device=opt.device)
    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'std.npy'))
    val_split_file = pjoin(opt.data_root, 'val.txt')
    opt.text_dir = pjoin(opt.data_root, 'texts')
    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_dataset = Text2MotionDataset(opt, mean, std, val_split_file)
    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file)
    opt.batch_size = 32
    opt.num_quantizers = 6
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)
    transformotion = Transformotion(code_dim=opt.code_dim, vq_model=vq_model, opt=opt)
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    result_dir = pjoin('/root/autodl-tmp/generation', opt.dataset_name, opt.name)
    joints_dir = pjoin(result_dir, 'joints')
    animation_dir = pjoin(result_dir, 'animations')
    
    def inv_transform(data):
        return data * std + mean
    converter = Joint2BVHConvertor()
    for i, (captions, motions, lengths) in enumerate(train_loader):
        print(f"Batch {i+1}")
        print("Captions:", len(captions))

        print("Motions:", motions.shape)
        print("Lengths:", lengths)
        # motion_idx, motion_emb = vq_model.encode(motions)
        
        # print(f"motion_emb {motion_idx.shape},", motion_idx)
        print("="*50)
        motions = motions.detach().float().to(opt.device)
        embedded_ids, som_pos, labels = embeddings.mix_embedding_ids(captions, motions)
        transformotion(embedded_ids, som_pos, labels)
        m_length = [20] * len(captions)
        r = 1
        mids = transformotion.generate(captions[0], m_length[0], 1, 1)
        motion_code_id = embeddings.get_code_idx_from_token_ids(mids)
        gathered_ids = repeat(motion_code_id, 'b n -> b n d', d=opt.num_quantizers)
        # print(f"+++++++++++++++++++++transformotion.token_embed_weight shape: {transformotion.token_embed_weight[-1].shape}++++gathered_ids shape{gathered_ids.shape}", mids)
        pred_motions = vq_model.forward_decoder(gathered_ids)


        pred_motions = pred_motions.detach().cpu().numpy()

        data = inv_transform(pred_motions)
        # print("+++++++=data", data)
        for k, (caption, joint_data)  in enumerate(zip(captions, data)):
            print("---->Sample %d: %s %d"%(k, caption, m_length[k]))
            animation_path = pjoin(animation_dir, str(k))
            joint_path = pjoin(joints_dir, str(k))

            os.makedirs(animation_path, exist_ok=True)
            os.makedirs(joint_path, exist_ok=True)

            joint_data = joint_data[:m_length[k]]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()

            bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.bvh"%(k, r, m_length[k]))
            _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)

            bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d.bvh" % (k, r, m_length[k]))
            _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)


            save_path = pjoin(animation_path, "sample%d_repeat%d_len%d.mp4"%(k, r, m_length[k]))
            ik_save_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.mp4"%(k, r, m_length[k]))

            plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title=caption, fps=20)
            plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)
            np.save(pjoin(joint_path, "sample%d_repeat%d_len%d.npy"%(k, r, m_length[k])), joint)
            np.save(pjoin(joint_path, "sample%d_repeat%d_len%d_ik.npy"%(k, r, m_length[k])), ik_joint)
        # for i, (caption) in enumerate(captions):
        #     if (i == 0):
        #         embeddings.mix_embedding(caption, motions[i])
        #         break
            # mix_embedding(caption, motion=motions[i], vq_model=vq_model)
        if (i == 1): break
            

    
