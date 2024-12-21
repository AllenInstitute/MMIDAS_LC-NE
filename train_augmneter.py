import argparse
import os
from mmidas.vaegan import vae_gan
from mmidas.utils.config_tools import get_paths
from mmidas.utils.data_tools import load_data, get_loaders
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--z_dim", default=15, type=int, help="latent dimension")
parser.add_argument("--noise_dim", default=3, type=int, help="noise dimension")
parser.add_argument("--alpha", default=2, type=int,  help="triple loss parameter")
parser.add_argument("--n_gene", default=0., type=int, help="number of genes")
parser.add_argument("--n_epoch", default=10000, type=int, help="Number of epochs to train")
parser.add_argument("--fc_dim", default=100, type=int, help="number of nodes at the hidden layers")
parser.add_argument("--batch_size", default=512, type=int, help="batch size")
parser.add_argument("--affine", default=False, type=bool, help="affine transformation in the batch normalization")
parser.add_argument("--momentum",  default=0.01, type=float, help="momentum for batch normalization")
parser.add_argument("--lr", default=.001, type=float, help="learning rate")
parser.add_argument("--p_drop", default=0.5, type=float, help="input probability of dropout")
parser.add_argument("--intial_w", default=False, type=bool, help="hard encoding")
parser.add_argument("--device", default=None, type=int, help="gpu device, use None for cpu")
parser.add_argument("--toml_file", default='pyproject.toml', type=str, help="the project toml file")


def main(z_dim, noise_dim, alpha, n_gene, n_epoch, fc_dim, batch_size, affine, lr, p_drop, momentum, intial_w, device, toml_file):

    config = get_paths(toml_file=toml_file)
    data_file = config['paths']['main_dir'] / config['paths']['data_path'] / config['data']['anndata_file']
    gene_file = config['paths']['main_dir'] / config['paths']['data_path'] / config['data']['hvg_file']
    
    saving_folder = config['paths']['main_dir'] / config['paths']['models']
    os.makedirs(saving_folder, exist_ok=True)
    saving_folder = str(saving_folder)


    data = load_data(file=data_file, gene_file=gene_file, n_gene=n_gene) 

    augmenter = vae_gan(saving_folder=saving_folder, device=device)
    
    data_loader, _, _, _, _, _ = get_loaders(x=data['log1p'], batch_size=batch_size, additional_val=False)

    augmenter.init_model(
                        input_dim=data['log1p'].shape[1], 
                        z_dim=z_dim, 
                        noise_dim=noise_dim, 
                        fc_dim=fc_dim, 
                        x_drop=p_drop, 
                        affine=affine, 
                        momentum=momentum,
                        )


    augmenter.train(
                    dataloader=data_loader, 
                    n_epoch=n_epoch, 
                    lr=lr, 
                    alpha=alpha, 
                    lam = [1, 0.5, .1, .5], 
                    )
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
