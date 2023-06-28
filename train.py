"""Training procedure for NICE.
"""

import argparse
import torch, torchvision
import numpy as np
import nice, utils
import funcoes_treino_lstm

def main(args):
    device = torch.device("cuda:0")

    # model hyperparameters
    dataset = args.dataset
    batch_size = args.batch_size
    latent = args.latent
    max_iter = args.max_iter
    sample_size = args.sample_size
    coupling = 4
    mask_config = 1.
    
    
    #dataset parameters
    category = args.category
    fold_idx = args.fold_idx
    

    # optimization hyperparameters
    lr = args.lr
    momentum = args.momentum
    decay = args.decay

    zca = None
    mean = None
    
    if dataset == 'mine':
        data = funcoes_treino_lstm.Mhad(split='train', fold_idx=fold_idx,  category=category)
        dado = data.get_dataset()
        mean = dado.mean()
        (full_dim, mid_dim, hidden) = (1 * 50 * 3, 1, 100)
        da = torch.Tensor(dado)
        trainloader = torch.utils.data.DataLoader(da,batch_size=1, shuffle=True, num_workers=2)
        
    if latent == 'normal':
        prior = torch.distributions.Normal(
            torch.tensor(0.).to(device), torch.tensor(1.).to(device))
    elif latent == 'logistic':
        prior = utils.StandardLogistic()

    filename = '%s_' % dataset \
             + 'bs%d_' % batch_size \
             + '%s_' % latent \
             + 'cp%d_' % coupling \
             + 'md%d_' % mid_dim \
             + 'hd%d_' % hidden

    flow = nice.NICE(prior=prior, 
                coupling=coupling, 
                in_out_dim=full_dim, 
                mid_dim=mid_dim, 
                hidden=hidden, 
                mask_config=mask_config).to(device)
    optimizer = torch.optim.Adam(
        flow.parameters(), lr=lr, betas=(momentum, decay), eps=1e-4)

    total_iter = 0
    train = True
    running_loss = 0
    
    while train:
        for _, data in enumerate(trainloader, 1):
            flow.train()    # set to training mode
            if total_iter == max_iter:
                train = False
                break

            total_iter += 1
            optimizer.zero_grad()    # clear gradient tensors

            inputs = data
            inputs = utils.prepare_data(
                inputs, dataset, zca=zca, mean=mean).to(device)

            # log-likelihood of input minibatch
            loss = -flow(inputs).mean()
            running_loss += float(loss)

            # backprop and update parameters
            loss.backward()
            optimizer.step()

            if total_iter % 1000 == 0:
                mean_loss = running_loss / 1000
                bit_per_dim = (mean_loss + np.log(256.) * full_dim) \
                            / (full_dim * np.log(2.))
                print('iter %s:' % total_iter, 
                    'loss = %.3f' % mean_loss, 
                    'bits/dim = %.3f' % bit_per_dim)
                running_loss = 0.0

                flow.eval()        # set to inference mode
                generated = flow.sample(size=500).cpu().detach().numpy()
                arq = open(f'generated/category-{category}-fold-{fold_idx}.npy', 'wb')
                np.save(arq, generated)
    flow.eval()        # set to inference mode
    generated = flow.sample(size=500).cpu().detach().numpy()
    arq = open(f'generated/category-{category}-fold-{fold_idx}.npy', 'wb')
    np.save(arq, generated)
   
    print('Finished training!')

    torch.save({
        'total_iter': total_iter, 
        'model_state_dict': flow.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict(), 
        'dataset': dataset, 
        'batch_size': batch_size, 
        'latent': latent, 
        'coupling': coupling, 
        'mid_dim': mid_dim, 
        'hidden': hidden, 
        'mask_config': mask_config}, 
        './models/' + filename +'iter%d.tar' % total_iter)

    print('Checkpoint Saved')
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('MNIST NICE PyTorch implementation')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=200)
    parser.add_argument('--latent',
                        help='latent distribution.',
                        type=str,
                        default='logistic')
    parser.add_argument('--max_iter',
                        help='maximum number of iterations.',
                        type=int,
                        default=25000)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)
    parser.add_argument('--momentum',
                        help='beta1 in Adam optimizer.',
                        type=float,
                        default=0.9)
    parser.add_argument('--decay',
                        help='beta2 in Adam optimizer.',
                        type=float,
                        default=0.999)
    parser.add_argument('--fold_idx',
                        help='fold to train.',
                        type=int,
                        default=0)
    parser.add_argument('--category',
                        help='category o train.',
                        type=int,
                        default=0)
    args = parser.parse_args()

    main(args)
