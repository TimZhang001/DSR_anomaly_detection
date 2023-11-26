import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary
from data_loader import TrainWholeImageDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import os


class VectorQuantizerEMA(nn.Module):
    # Source for the VectorQuantizerEMA module: https://github.com/zalandoresearch/pytorch-vq-vae
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim  = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay   = decay
        self._epsilon = epsilon

    def get_quantized(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings        = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        quantized = inputs + (quantized - inputs).detach()

        return quantized.permute(0, 3, 1, 2).contiguous()

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings



class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class EncoderBot(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(EncoderBot, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class EncoderTop(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(EncoderTop, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._residual_stack(x)
        return x


class DecoderBot(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(DecoderBot, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=3,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)

class DiscreteLatentModel(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim,
                 commitment_cost, decay=0, test=False):
        # def __init__(self, num_embeddings=512, embedding_dim=128, commitment_cost=0.25, decay=0):
        super(DiscreteLatentModel, self).__init__()
        self.test       = test
        self._encoder_b = EncoderBot(3, num_hiddens,
                                     num_residual_layers,
                                     num_residual_hiddens)
        
        self._encoder_t = EncoderTop(num_hiddens, num_hiddens,
                                     num_residual_layers,
                                     num_residual_hiddens)

        self._pre_vq_conv_bot = nn.Conv2d(in_channels=num_hiddens + embedding_dim,
                                          out_channels=embedding_dim,
                                          kernel_size=1,
                                          stride=1)

        self._pre_vq_conv_top = nn.Conv2d(in_channels=num_hiddens,
                                          out_channels=embedding_dim,
                                          kernel_size=1,
                                          stride=1)

        self._vq_vae_top = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)

        self._vq_vae_bot = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)

        self._decoder_b = DecoderBot(embedding_dim*2,
                                     num_hiddens,
                                     num_residual_layers,
                                     num_residual_hiddens)


        self.upsample_t = nn.ConvTranspose2d(embedding_dim, embedding_dim, 4, stride=2, padding=1)

    def generate_fake_anomalies(self, features, memory_weight, strength=0.2):
        random_embeddings = torch.zeros((features.shape[0],features.shape[2]*features.shape[3], memory_weight.shape[1]))
        inputs            = features.permute(0, 2, 3, 1).contiguous()

        for k in range(features.shape[0]):
            memory_torch = memory_weight
            flat_input   = inputs[k].view(-1, memory_torch.shape[1])

            # 计算与码本权重的距离
            distances_b = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                        + torch.sum(memory_torch ** 2, dim=1)
                        - 2 * torch.matmul(flat_input, memory_torch.t()))

            # 选择距离最小的topk个(0.2~1)
            percentage_vectors   = strength
            topk                 = max(1, min(int(percentage_vectors * memory_torch.shape[0]) + 1, memory_torch.shape[0] - 1))
            values, topk_indices = torch.topk(distances_b, topk, dim=1, largest=False)
            
            # 去除距离最小的5%的点(4096*0.05=204)
            topk_indices         = topk_indices[:, int(memory_torch.shape[0] * 0.05):]
            topk                 = topk_indices.shape[1]

            # 从中随机选择1024个
            random_indices_hik   = torch.randint(topk, size=(topk_indices.shape[0],))
            random_indices_t     = topk_indices[torch.arange(random_indices_hik.shape[0]),random_indices_hik]
            random_embeddings[k] = memory_torch[random_indices_t,:]
        
        random_embeddings        = random_embeddings.reshape((random_embeddings.shape[0],
                                                            features.shape[2],features.shape[3],random_embeddings.shape[2]))
        random_embeddings_tensor = random_embeddings.permute(0,3,1,2).cuda()

        return random_embeddings_tensor
    
    def do_fake_forward(self, x, strength):
        #Encoder Hi
        enc_b = self._encoder_b(x)

        #Encoder Lo -- F_Lo
        enc_t = self._encoder_t(enc_b)
        zt    = self._pre_vq_conv_top(enc_t)

        # Quantize F_Lo with K_Lo
        quantized_t = self.generate_fake_anomalies(zt, self._vq_vae_top._embedding.weight, strength=strength)
        
        # Upsample Q_Lo
        up_quantized_t = self.upsample_t(quantized_t)

        # Concatenate and transform the output of Encoder_Hi and upsampled Q_lo -- F_Hi
        feat = torch.cat((enc_b, up_quantized_t), dim=1)
        zb   = self._pre_vq_conv_bot(feat)

        # Quantize F_Hi with K_Hi
        quantized_b = self.generate_fake_anomalies(zb, self._vq_vae_bot._embedding.weight, strength=strength)

        # Concatenate Q_Hi and Q_Lo and input it into the General appearance decoder
        quant_join = torch.cat((up_quantized_t, quantized_b), dim=1)
        recon_fin  = self._decoder_b(quant_join)

        #return loss_b, loss_t, recon_fin, encodings_t, encodings_b, quantized_t, quantized_b
        return recon_fin, quantized_t, quantized_b

    def forward(self, x):
        #Encoder Hi
        enc_b = self._encoder_b(x)

        #Encoder Lo -- F_Lo
        enc_t = self._encoder_t(enc_b)
        zt    = self._pre_vq_conv_top(enc_t)

        # Quantize F_Lo with K_Lo
        loss_t, quantized_t, perplexity_t, encodings_t = self._vq_vae_top(zt)
        
        # Upsample Q_Lo
        up_quantized_t = self.upsample_t(quantized_t)

        # Concatenate and transform the output of Encoder_Hi and upsampled Q_lo -- F_Hi
        feat = torch.cat((enc_b, up_quantized_t), dim=1)
        zb   = self._pre_vq_conv_bot(feat)

        # Quantize F_Hi with K_Hi
        loss_b, quantized_b, perplexity_b, encodings_b = self._vq_vae_bot(zb)

        # Concatenate Q_Hi and Q_Lo and input it into the General appearance decoder
        quant_join = torch.cat((up_quantized_t, quantized_b), dim=1)
        recon_fin  = self._decoder_b(quant_join)

        #return loss_b, loss_t, recon_fin, encodings_t, encodings_b, quantized_t, quantized_b
        return loss_b, loss_t, recon_fin, quantized_t, quantized_b
    

if __name__=="__main__":

    model = DiscreteLatentModel(num_hiddens=128, num_residual_layers=2, num_residual_hiddens=64,
                                num_embeddings=4096, embedding_dim=128, commitment_cost=0.25, decay=0.99)
    
    run_name_pre = 'vq_model_pretrained_128_4096'
    mvtec_path   = '/raid/zhangss/dataset/ADetection/mvtecAD/'
    obj_name     = 'transistor'

    model.cuda()
    model.load_state_dict(torch.load("./checkpoints/" + run_name_pre + ".pckl", map_location='cuda:0'))
    model.eval()

    dataset    = TrainWholeImageDataset(mvtec_path + obj_name + "/train/good/", resize_shape=[256, 256], perlin_augment=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    for i_batch, sample_batched in enumerate(dataloader):
        in_image     = sample_batched["image"].cuda()
        anomaly_mask = sample_batched["mask"].cuda()

        _, _, recon_fin, quantized_t, quantized_b = model(in_image) 
        image_err = torch.abs(in_image - recon_fin)

        save_path = "DebugImage"
        os.makedirs(save_path, exist_ok=True)
        utils.save_image(in_image,  os.path.join(save_path, str(i_batch).zfill(4) + "_in_image.png"),  nrow=4)
        utils.save_image(recon_fin, os.path.join(save_path, str(i_batch).zfill(4) + "_rec_image.png"), nrow=4)
        utils.save_image(image_err, os.path.join(save_path, str(i_batch).zfill(4) + "_err_image.png"), nrow=4, normalize=True)
        save_quantized_t = torch.mean(quantized_t, keepdim=True, dim=1)
        save_quantized_b = torch.mean(quantized_b, keepdim=True, dim=1)
        utils.save_image(save_quantized_t, os.path.join(save_path, str(i_batch).zfill(4) + "_quantized_t.png"), nrow=4, normalize=True)
        utils.save_image(save_quantized_b, os.path.join(save_path, str(i_batch).zfill(4) + "_quantized_b.png"), nrow=4, normalize=True)

        for idex in range(2, 10):
            strength = 0.1 * idex
            recon_fin_fake, quantized_t_fake, quantized_b_fake = model.do_fake_forward(in_image, strength=strength)
            utils.save_image(recon_fin_fake, os.path.join(save_path, str(i_batch).zfill(4) + "_rec_image_fake_" + str(strength) + ".png"), nrow=4)


       