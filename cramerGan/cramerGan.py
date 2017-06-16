import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.nn.utils import clip_grad_norm
from .utils import plot_img, plot_scalar, save_images, to_device

def Critic(netD, real, fake2):
    net_real = netD(real)
    return torch.norm(net_real - netD(fake2), p=2, dim=1) - \
           torch.norm(net_real, p =2,  dim=1)

def calc_gradient_penalty(netD, real_data, fake_data):
    #print real_data.size()
    batch_size = real_data.size()[0]
    one_list = [1]*len(list(real_data.size()[1:]))
    alpha = torch.rand(batch_size,*one_list)
    
    alpha = alpha.expand_as(real_data)
    alpha = to_device(alpha, netD.device_id, False)
    interpolates = alpha * real_data.data + (1 - alpha) * fake_data.data
    interpolates =  to_device(interpolates, netD.device_id)
    #interpolates.requires_grad = True
    disc_interpolates = Critic(netD, interpolates, fake_data)
    
    grad_outputs = to_device(torch.ones(disc_interpolates.size()), netD.device_id, False)
    
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=grad_outputs,
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty

def cramer_gans(netG, netD, real, z1, z2, args):
    real = to_device(real, netD.device_id)
    fake1, fake2 = netG(z1), netG(z2)
    disc_real  = netD(real)
    disc_fake1 = netD(fake1)
    disc_fake2 = netD(fake2)

    #if update_G:
    gen_loss = torch.mean(
        torch.norm(disc_real - disc_fake1,  p =2, dim = 1)
        + torch.norm(disc_real - disc_fake2, p =2, dim = 1)
        - torch.norm(disc_fake1 - disc_fake2, p =2, dim = 1)
    )
    #    return gen_loss
    #else:    
    surrogate = torch.mean( Critic(netD, real, fake2) - 
                            Critic(netD, fake1, fake2))
    
    grad_penalty = calc_gradient_penalty(netD, real, fake1)
    
    disc_loss = -surrogate + grad_penalty * args.gp_lambda
    return gen_loss, disc_loss

def train_gans(x_sampler, model_root, mode_name, netG, netD, args):

    optimizerD = optim.Adam(netD.parameters(), lr= args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    optimizerG = optim.Adam(netG.parameters(), lr= args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)

    #optimizerD = optim.RMSprop(netD.parameters(), lr= args.lr,  weight_decay=args.weight_decay)
    #optimizerG = optim.RMSprop(netG.parameters(), lr= args.lr,  weight_decay=args.weight_decay)
    model_folder = os.path.join(model_root, mode_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    D_weightspath = os.path.join(model_folder, 'd_weights.pth')
    G_weightspath = os.path.join(model_folder, 'g_weights.pth')
    if args.reuse_weigths == 1:
        if os.path.exists(D_weightspath):
            weights_dict = torch.load(D_weightspath,map_location=lambda storage, loc: storage)
            netD.load_state_dict(weights_dict)# 12)
            print('reload weights from {}'.format(D_weightspath))
        
        if os.path.exists(G_weightspath):
            weights_dict = torch.load(G_weightspath,map_location=lambda storage, loc: storage)
            netG.load_state_dict(weights_dict)# 12)
            print('reload weights from {}'.format(G_weightspath))
    d_loss_plot = plot_scalar(name = "d_loss", env= mode_name, rate = args.display_freq)
    g_loss_plot = plot_scalar(name = "g_loss", env= mode_name, rate = args.display_freq)

    def z_sampler(batch_size, noise_dim):
        z = to_device(torch.randn(batch_size, noise_dim), netG.device_id)
        return z

    for batch_count in range(args.maxepoch):
        for _ in range(args.ncritic):
            # (1) Update D network
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True   # they are set to False below in netG update
                #p.data.clamp_(-1, 1)
            netD.zero_grad()
            real = x_sampler()
            batch_size = real.shape[0]

            z1   = z_sampler(batch_size, args.noise_dim)
            z2   = z_sampler(batch_size, args.noise_dim)

            g_loss, d_loss = cramer_gans(netG, netD, real, z1, z2, args)
            d_loss_plot.plot(d_loss.cpu().data.numpy().mean())
            
            d_loss.backward(retain_variables=True)
            optimizerD.step()
            
        # (2) Update G network
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        
        #g_loss, d_loss = cramer_gans(netG, netD, real, z1, z2, args)
        g_loss_plot.plot(g_loss.cpu().data.numpy().mean())
        netG.zero_grad()
        g_loss.backward()
        optimizerG.step()
        #real = x_sampler()
        #batch_size = real.shape[0]
        #z1   = z_sampler(batch_size, args.noise_dim)
        #z2   = z_sampler(batch_size, args.noise_dim)
        #g_loss = cramer_gans(True, netG, netD, real, z1, z2, args)
        #torch.nn.utils.clip_grad_norm(netG.parameters(), args.grad_clip)
        
        # Calculate dev loss and generate samples every 100 iters
        if batch_count % args.display_freq == 0:
            print('save tmp images, :)')
            z1   = z_sampler(batch_size, args.noise_dim)
            samples = netG(z1).cpu().data.numpy()
            imgs = save_images(
                        samples,
                        os.path.join(args.save_folder,'samples_{}.png'.format(batch_count) ),save=False,dim_ordering = 'th'
                        )
            print(samples.shape)
            plot_img(X=imgs, win='sample_img', env=mode_name)
            
            true_imgs = save_images(real, save=False,dim_ordering = 'th')
            plot_img(X=true_imgs, win='real_img', env=mode_name)
        
        if batch_count % args.save_freq==0:
            D_cur_weights = netD.state_dict()
            G_cur_weights = netG.state_dict()
            torch.save(D_cur_weights, D_weightspath)
            torch.save(G_cur_weights, G_weightspath)
            print('save weights to {} and {}'.format(D_weightspath, G_weightspath))

