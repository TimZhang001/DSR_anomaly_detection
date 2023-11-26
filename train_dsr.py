import torch
import torch.nn.functional as F
from data_loader import TrainWholeImageDataset, MVTecImageAnomTrainDataset
from torch.utils.data import DataLoader
from torch import optim
import os
import numpy as np
from dsr_model import SubspaceRestrictionModule, ImageReconstructionNetwork, AnomalyDetectionModule, UpsamplingModule
from discrete_model import DiscreteLatentModel
import sys
from loss import FocalLoss
from sklearn.metrics import roc_auc_score, average_precision_score
from data_loader_test import TestMVTecDataset
import time
from torchvision import transforms, utils
from keras_sequential_ascii import keras2ascii
import matplotlib.pyplot as plt

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def generate_fake_anomalies_joined(features, embeddings, memory_torch_original, mask, diversity=1.0, strength=None):
    random_embeddings = torch.zeros((embeddings.shape[0],embeddings.shape[2]*embeddings.shape[3], memory_torch_original.shape[1]))
    inputs            = features.permute(0, 2, 3, 1).contiguous()

    for k in range(embeddings.shape[0]):
        memory_torch = memory_torch_original
        flat_input   = inputs[k].view(-1, memory_torch.shape[1])

        # 计算与码本权重的距离
        distances_b = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(memory_torch ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, memory_torch.t()))

        # 选择距离最小的topk个(0.2~1)
        percentage_vectors   = strength[k]
        topk                 = max(1, min(int(percentage_vectors * memory_torch.shape[0]) + 1, memory_torch.shape[0] - 1))
        values, topk_indices = torch.topk(distances_b, topk, dim=1, largest=False)
        
        # 去除距离最小的5%的点(4096*0.05=204)
        topk_indices         = topk_indices[:, int(memory_torch.shape[0] * 0.05):]
        topk                 = topk_indices.shape[1]

        # 从中随机选择1024个
        random_indices_hik   = torch.randint(topk, size=(topk_indices.shape[0],))
        random_indices_t     = topk_indices[torch.arange(random_indices_hik.shape[0]),random_indices_hik]
        random_embeddings[k] = memory_torch[random_indices_t,:]
    random_embeddings       = random_embeddings.reshape((random_embeddings.shape[0],embeddings.shape[2],embeddings.shape[3],random_embeddings.shape[2]))
    random_embeddings_tensor = random_embeddings.permute(0,3,1,2).cuda()

    down_ratio_y = int(mask.shape[2]/embeddings.shape[2])
    down_ratio_x = int(mask.shape[3]/embeddings.shape[3])
    anomaly_mask = torch.nn.functional.max_pool2d(mask, (down_ratio_y, down_ratio_x)).float()

    anomaly_embedding = anomaly_mask * random_embeddings_tensor + (1.0 - anomaly_mask) * embeddings

    return anomaly_embedding

def train_upsampling_module(model, sub_res_model_hi, sub_res_model_lo, model_decode, decoder_seg, obj_name, mvtec_path, out_path, lr, batch_size, epochs):
    anom_par = 0.2
    run_name = 'dsr_' + str(lr) + '_' + str(epochs) + '_bs' + str(batch_size) + "_anom_par" + str(anom_par) + "_"

    embedding_dim = 128
    model.eval()
    sub_res_model_hi.eval()
    sub_res_model_lo.eval()
    decoder_seg.eval()
    model_decode.eval()

    model_upsample = UpsamplingModule(embedding_size=embedding_dim)
    model_upsample.cuda()
    model_upsample.train()


    optimizer = torch.optim.Adam([
                                  {"params": model_upsample.parameters(), "lr": lr}
                                 ])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[int(epochs*0.80), int(epochs*0.90)],gamma=0.2, last_epoch=-1)

    loss_focal = FocalLoss()

    dataset    = MVTecImageAnomTrainDataset(mvtec_path + obj_name + "/train/good/", resize_shape=[256, 256])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    n_iter = 0.0

    segment_loss_avg = 0.0

    for epoch in range(epochs//2):
        start_time = time.time()
        for i_batch, sample_batched in enumerate(dataloader):

            input_image_aug = sample_batched["augmented_image"].cuda()
            anomaly_mask    = sample_batched["anomaly_mask"].cuda()

            optimizer.zero_grad()

            loss_b, loss_t, data_recon, embeddings_t, embeddings = model(input_image_aug)

            data_recon = data_recon.detach()
            embeddings = embeddings.detach()
            embeddings_t = embeddings_t.detach()

            embedder     = model._vq_vae_bot
            embedder_top = model._vq_vae_top

            anomaly_embedding_copy     = embeddings.clone()
            anomaly_embedding_top_copy = embeddings_t.clone()
            recon_feat, recon_embeddings, _ = sub_res_model_hi(anomaly_embedding_copy, embedder)
            recon_feat_top, recon_embeddings_top, loss_b_top = sub_res_model_lo(anomaly_embedding_top_copy,
                                                                                embedder_top)

            up_quantized_recon_t = model.upsample_t(recon_embeddings_top)
            quant_join           = torch.cat((up_quantized_recon_t, recon_embeddings), dim=1)
            recon_image_recon    = model_decode(quant_join)

            ################################################
            up_quantized_embedding_t = model.upsample_t(embeddings_t)
            quant_join_real = torch.cat((up_quantized_embedding_t, embeddings), dim=1)
            recon_image     = model._decoder_b(quant_join_real)

            out_mask     = decoder_seg(recon_image_recon, recon_image)
            out_mask_sm  = torch.softmax(out_mask, dim=1)
            refined_mask = model_upsample(recon_image_recon, recon_image, out_mask_sm)
            refined_mask_sm = torch.softmax(refined_mask, dim=1)

            segment_loss = loss_focal(refined_mask_sm, anomaly_mask)

            loss = segment_loss
            loss.backward()
            optimizer.step()

            segment_loss_avg = segment_loss_avg * 0.95 + 0.05 * segment_loss.item()

            n_iter +=1

        scheduler.step()

        if epoch % 5 == 0:
            save_path = os.path.join(out_path, run_name, "checkpoints")
            torch.save(model_upsample.state_dict(), os.path.join(save_path + "upsample_"+obj_name+".pckl"))


def train_on_device(obj_names, mvtec_path, out_path, lr, batch_size, epochs):
    run_name_pre = 'vq_model_pretrained_128_4096'
    num_hiddens  = 128
    num_residual_hiddens = 64
    num_residual_layers  = 2
    embedding_dim   = 128
    num_embeddings  = 4096
    commitment_cost = 0.25
    decay           = 0.99
    anom_par        = 0.2

    # Load the pretrained discrete latent model used.
    model = DiscreteLatentModel(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim,
                                commitment_cost, decay)
    model.cuda()
    model.load_state_dict(torch.load("./checkpoints/" + run_name_pre + ".pckl", map_location='cuda:0'))
    model.eval()

    # Modules using the codebooks K_hi and K_lo for feature quantization
    embedder_hi = model._vq_vae_bot
    embedder_lo = model._vq_vae_top

    for obj_name in obj_names:
        run_name = 'dsr_'+str(lr)+'_'+str(epochs)+'_bs'+str(batch_size)+"_anom_par"+str(anom_par)+"_"

        # Define the subspace restriction modules - Encoder decoder networks
        sub_res_model_lo = SubspaceRestrictionModule(embedding_size=embedding_dim)
        sub_res_model_hi = SubspaceRestrictionModule(embedding_size=embedding_dim)
        sub_res_model_lo.cuda()
        sub_res_model_hi.cuda()

        # Define the anomaly detection module - UNet-based network
        decoder_seg = AnomalyDetectionModule(embedding_size=embedding_dim)
        decoder_seg.cuda()
        decoder_seg.apply(weights_init)


        # Image reconstruction network reconstructs the image from discrete features.
        # It is trained for a specific object
        model_decode = ImageReconstructionNetwork(embedding_dim * 2,
                                                  num_hiddens,
                                                  num_residual_layers,
                                                  num_residual_hiddens)
        model_decode.cuda()
        model_decode.apply(weights_init)

        optimizer = torch.optim.Adam([
                                      {"params": sub_res_model_lo.parameters(), "lr": lr},
                                      {"params": sub_res_model_hi.parameters(), "lr": lr},
                                      {"params": model_decode.parameters(), "lr": lr},
                                      {"params": decoder_seg.parameters(), "lr": lr}])

        scheduler  = optim.lr_scheduler.MultiStepLR(optimizer,[int(epochs*0.80)],gamma=0.1, last_epoch=-1)
        loss_focal = FocalLoss()

        dataset    = TrainWholeImageDataset(mvtec_path + obj_name + "/train/good/", resize_shape=[256, 256], perlin_augment=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        n_train    = len(dataset)
        n_iter     = 0.0
        start_time = time.time()
        for epoch in range(epochs):
            print("Epoch ", epoch, " of ", epochs, " time: ", time.time() - start_time)
            for i_batch, sample_batched in enumerate(dataloader):

                in_image     = sample_batched["image"].cuda()
                anomaly_mask = sample_batched["mask"].cuda()
                optimizer.zero_grad()

                with torch.no_grad():
                    
                    # 0.2~1.0
                    anomaly_strength_lo = (torch.rand(in_image.shape[0]) * (1.0-anom_par) + anom_par).cuda()
                    anomaly_strength_hi = (torch.rand(in_image.shape[0]) * (1.0-anom_par) + anom_par).cuda()

                    # -----------------------------------------------------------------------------------
                    # Extract features from the discrete model
                    enc_b = model._encoder_b(in_image)    # in 8*3*256*256 out 8*128*64*64  lowlevel
                    enc_t = model._encoder_t(enc_b)       # in 8*128*64*64 out 8*128*32*32  highlevel
                    zt    = model._pre_vq_conv_top(enc_t) # in 8*128*32*32 out 8*128*32*32

                    # Quantize the extracted features
                    loss_t, quantized_t_real, perplexity_t, encodings_t = embedder_lo(zt)

                    # Generate feature-based anomalies on F_lo
                    anomaly_embedding_lo = generate_fake_anomalies_joined(zt, quantized_t_real,
                                                                          embedder_lo._embedding.weight,
                                                                          anomaly_mask, strength=anomaly_strength_lo)

                    # Upsample the extracted quantized features and the quantized features augmented with anomalies
                    up_quantized_t      = model.upsample_t(anomaly_embedding_lo)
                    up_quantized_t_real = model.upsample_t(quantized_t_real)
                    feat      = torch.cat((enc_b, up_quantized_t), dim=1)
                    feat_real = torch.cat((enc_b, up_quantized_t_real), dim=1)
                    zb        = model._pre_vq_conv_bot(feat)
                    zb_real   = model._pre_vq_conv_bot(feat_real)
                    # Quantize the upsampled features - F_hi
                    loss_b, quantized_b,      perplexity_b, encodings_b = embedder_hi(zb)
                    loss_b, quantized_b_real, perplexity_b, encodings_b = embedder_hi(zb_real)

                    # Generate feature-based anomalies on F_hi 
                    # fake - fake both
                    anomaly_embedding_hi = generate_fake_anomalies_joined(zb, quantized_b,
                                                                          embedder_hi._embedding.weight, 
                                                                          anomaly_mask, strength=anomaly_strength_hi)

                    # real - fake top
                    anomaly_embedding_hi_usebot = generate_fake_anomalies_joined(zb_real, quantized_b_real,
                                                                                 embedder_hi._embedding.weight,
                                                                                 anomaly_mask, strength=anomaly_strength_hi)
                    
                    # 
                    use_both = torch.randint(0, 2,(in_image.shape[0],1,1,1)).cuda().float()
                    use_lo   = torch.randint(0, 2,(in_image.shape[0],1,1,1)).cuda().float()
                    use_hi   = (1 - use_lo)

                    anomaly_embedding_lo_usebot = quantized_t_real
                    anomaly_embedding_hi_usetop = quantized_b_real
                    anomaly_embedding_lo_usetop = anomaly_embedding_lo
                    anomaly_embedding_hi_not_both =  use_hi * anomaly_embedding_hi_usebot + use_lo * anomaly_embedding_hi_usetop
                    anomaly_embedding_lo_not_both =  use_hi * anomaly_embedding_lo_usebot + use_lo * anomaly_embedding_lo_usetop
                    anomaly_embedding_hi = (anomaly_embedding_hi * use_both + anomaly_embedding_hi_not_both * (1.0 - use_both)).detach().clone()
                    anomaly_embedding_lo = (anomaly_embedding_lo * use_both + anomaly_embedding_lo_not_both * (1.0 - use_both)).detach().clone()

                    anomaly_embedding_hi_copy = anomaly_embedding_hi.clone()
                    anomaly_embedding_lo_copy = anomaly_embedding_lo.clone()

                # -----------------------------------------------------------------------------------
                # Restore the features to normality with the Subspace restriction modules
                # 量化离散 -> 连续/重构 -> 量化离散
                recon_feat_hi, recon_embeddings_hi, loss_b    = sub_res_model_hi(anomaly_embedding_hi_copy, embedder_hi)
                recon_feat_lo, recon_embeddings_lo, loss_b_lo = sub_res_model_lo(anomaly_embedding_lo_copy, embedder_lo)

                # Reconstruct the image from the anomalous features with the general appearance decoder
                up_quantized_anomaly_t = model.upsample_t(anomaly_embedding_lo)
                quant_join_anomaly     = torch.cat((up_quantized_anomaly_t, anomaly_embedding_hi), dim=1)
                recon_image_general    = model._decoder_b(quant_join_anomaly)


                # Reconstruct the image from the reconstructed features
                # with the object-specific image reconstruction module
                up_quantized_recon_t = model.upsample_t(recon_embeddings_lo)
                quant_join           = torch.cat((up_quantized_recon_t, recon_embeddings_hi), dim=1)
                recon_image_recon    = model_decode(quant_join)

                # Generate the anomaly segmentation map
                out_mask    = decoder_seg(recon_image_recon.detach(),recon_image_general.detach())
                out_mask_sm = torch.softmax(out_mask, dim=1)

                # Calculate losses
                loss_feat_hi = torch.nn.functional.mse_loss(recon_feat_hi, quantized_b_real.detach())
                loss_feat_lo = torch.nn.functional.mse_loss(recon_feat_lo, quantized_t_real.detach())
                loss_l2_recon_img = torch.nn.functional.mse_loss(in_image, recon_image_recon)
                total_recon_loss  = loss_feat_lo + loss_feat_hi + loss_l2_recon_img*10


                # Resize the ground truth anomaly map to closely match the augmented features
                down_ratio_x_hi = int(anomaly_mask.shape[3] / quantized_b.shape[3])
                anomaly_mask_hi = torch.nn.functional.max_pool2d(anomaly_mask, (down_ratio_x_hi, down_ratio_x_hi)).float()
                anomaly_mask_hi = torch.nn.functional.interpolate(anomaly_mask_hi, scale_factor=down_ratio_x_hi)
                down_ratio_x_lo = int(anomaly_mask.shape[3] / quantized_t_real.shape[3])
                anomaly_mask_lo = torch.nn.functional.max_pool2d(anomaly_mask, (down_ratio_x_lo, down_ratio_x_lo)).float()
                anomaly_mask_lo = torch.nn.functional.interpolate(anomaly_mask_lo, scale_factor=down_ratio_x_lo)
                anomaly_mask    = anomaly_mask_lo * use_both + (anomaly_mask_lo * use_lo + anomaly_mask_hi * use_hi) * (1.0 - use_both)


                # Calculate the segmentation loss with GT mask generated at low resolution.
                segment_loss = loss_focal(out_mask_sm, anomaly_mask)

                loss = segment_loss + total_recon_loss
                loss.backward()
                optimizer.step()

                if i_batch == 0:
                    print("Loss Focal: ", segment_loss.item())
                    print("Loss Recon: ", total_recon_loss.item())
                    save_path = os.path.join(out_path, run_name, "DebugImage") 
                    os.makedirs(save_path, exist_ok=True)
                    utils.save_image(in_image,            os.path.join(save_path, str(epoch).zfill(4) + "_in_image.png"), nrow=4)
                    utils.save_image(recon_image_recon,   os.path.join(save_path, str(epoch).zfill(4) + "_recon_image_recon.png"), nrow=4)
                    utils.save_image(recon_image_general, os.path.join(save_path, str(epoch).zfill(4) + "_recon_image_general.png"), nrow=4)            
                    utils.save_image(anomaly_mask,        os.path.join(save_path, str(epoch).zfill(4) + "_anomaly_mask.png"), nrow=4)
                    utils.save_image(out_mask_sm,         os.path.join(save_path, str(epoch).zfill(4) + "_out_mask_sm.png"), nrow=4)

                    save_quant_join_anomaly = torch.mean(quant_join_anomaly, dim=1)
                    save_quant_join         = torch.mean(quant_join, dim=1)
                    
                    plt.figure(figsize=(20, 8))
                    for k in range(save_quant_join_anomaly.shape[0]):
                        plt.subplot(2, save_quant_join_anomaly.shape[0], k+1)
                        plt.imshow(save_quant_join_anomaly[k].cpu().detach().numpy())
                        plt.axis('off')
                        plt.subplot(2, save_quant_join_anomaly.shape[0], k+1+save_quant_join_anomaly.shape[0])
                        plt.imshow(save_quant_join[k].cpu().detach().numpy())
                        plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_path, str(epoch).zfill(4) + "_quant_feature_wo_anomaly.png"))
   
                n_iter +=1

            scheduler.step()

            if (epoch+1) % 5 == 0:
                # Save models
                save_path = os.path.join(out_path, run_name, "checkpoints")
                torch.save(decoder_seg.state_dict(),      os.path.join(save_path + "anomaly_det_module_"+obj_name+".pckl"))
                torch.save(sub_res_model_lo.state_dict(), os.path.join(save_path + "subspace_restriction_lo_"+obj_name+".pckl"))
                torch.save(sub_res_model_hi.state_dict(), os.path.join(save_path + "subspace_restriction_hi_"+obj_name+".pckl"))
                torch.save(model_decode.state_dict(),     os.path.join(save_path + "image_recon_module_"+obj_name+".pckl"))

    return model, sub_res_model_hi, sub_res_model_lo, model_decode, decoder_seg

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id',    action='store', type=int,   default=0)
    parser.add_argument('--bs',        action='store', type=int,   default=8)
    parser.add_argument('--lr',        action='store', type=float, default=0.0002)
    parser.add_argument('--epochs',    action='store', type=int,   default=100)
    parser.add_argument('--gpu_id',    action='store', type=int,   default=0)
    parser.add_argument('--data_path', action='store', type=str,   default='/raid/zhangss/dataset/ADetection/mvtecAD/')
    parser.add_argument('--out_path',  action='store', type=str,   default='/home/zhangss/Tim.Zhang/ADetection/DSR_ECCV2022/output1')

    args = parser.parse_args()

    # Use: python train_dsr.py --gpu_id 0 --obj_id $i --lr 0.0002 --bs 8 --epochs 100 --data_path $BASE_PATH --out_path $OUT_PATH
    # BASE_PATH -- the base directory of mvtec
    # OUT_PATH -- where the trained models will be saved
    # i -- the index of the object class in the obj_batch list
    obj_batch = [['capsule'],
                 ['bottle'],
                 ['carpet'],
                 ['leather'],
                 ['pill'],
                 ['transistor'],
                 ['tile'],
                 ['cable'],
                 ['zipper'],
                 ['toothbrush'],
                 ['metal_nut'],
                 ['hazelnut'],
                 ['screw'],
                 ['grid'],
                 ['wood']
                 ]

    with torch.cuda.device(args.gpu_id):
        model, sub_res_model_hi, sub_res_model_lo, model_decode, decoder_seg = train_on_device(obj_batch[int(args.obj_id)],args.data_path, 
                                                                                               args.out_path, 
                                                                                               args.lr, 
                                                                                               args.bs, 
                                                                                               args.epochs)
        

        
        train_upsampling_module(model, sub_res_model_hi, sub_res_model_lo, model_decode, decoder_seg,
                                obj_batch[int(args.obj_id)], args.data_path, args.out_path, args.lr, args.bs, args.epochs)

