import argparse
import os
from mmidas.cplMixVAE import cpl_mixVAE
from mmidas.utils.config_tools import get_paths
from mmidas.utils.data_tools import load_data, get_loaders
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--n_categories", default=15, type=int, help="number of cell types")
parser.add_argument("--state_dim", default=3, type=int, help="state variable dimension")
parser.add_argument("--n_arm", default=2, type=int,  help="number of mixVAE arms for each modalities")
parser.add_argument("--temp",  default=1, type=float, help="gumbel-softmax temperature")
parser.add_argument("--tau",  default=.005, type=float, help="softmax temperature")
parser.add_argument("--beta",  default=1, type=float, help="KL regularization parameter")
parser.add_argument("--lam",  default=1, type=float, help="coupling factor")
parser.add_argument("--lam_pc",  default=1000, type=float, help="coupling factor for ref arm")
parser.add_argument("--ref_pc", default=False, type=bool, help="path of the data augmenter")
parser.add_argument("--latent_dim", default=10, type=int, help="latent dimension")
parser.add_argument("--n_epoch", default=10000, type=int, help="Number of epochs to train")
parser.add_argument("--n_epoch_p", default=10000, type=int, help="Number of epochs to train pruning algorithm")
parser.add_argument("--min_con", default=.99, type=float, help="minimum consensus")
parser.add_argument("--max_prun_it", default=14, type=int, help="maximum number of pruning iterations")
parser.add_argument("--n_aug_smp", default=0, type=int, help="number of augmented samples")
parser.add_argument("--fc_dim", default=100, type=int, help="number of nodes at the hidden layers")
parser.add_argument("--batch_size", default=512, type=int, help="batch size")
parser.add_argument("--variational", default=True, type=bool, help="enable variational mode")
parser.add_argument("--augmentation", default=False, type=bool, help="enable VAE-GAN augmentation")
parser.add_argument("--lr", default=.001, type=float, help="learning rate")
parser.add_argument("--n_gene", default=0., type=int, help="number of genes")
parser.add_argument("--p_drop", default=0.5, type=float, help="input probability of dropout")
parser.add_argument("--s_drop", default=0.0, type=float, help="state probability of dropout")
parser.add_argument("--n_run", default=1, type=int, help="number of the experiment")
parser.add_argument("--hard", default=False, type=bool, help="hard encoding")
parser.add_argument("--pre_trained_model", default='', type=str, help="pre-trained model")
parser.add_argument("--n_prun_c", default=0, type=int, help="number of prunned categories")
parser.add_argument("--training_mode", default='MSE', type=str, help="mode of the reconstruction loss: MSE or ZINB")
parser.add_argument("--device", default=None, type=int, help="gpu device, use None for cpu")
parser.add_argument("--toml_file", default='pyproject.toml', type=str, help="the project toml file")


def main(n_categories, 
        n_arm, 
        state_dim, 
        latent_dim, 
        fc_dim, 
        n_epoch, 
        n_epoch_p, 
        min_con,
        max_prun_it, 
        batch_size, 
        n_aug_smp,
        p_drop, 
        s_drop, 
        lr, 
        temp, 
        n_run, 
        device, 
        hard, 
        tau, 
        variational, 
        ref_pc, 
        augmentation, 
        n_gene, 
        lam, 
        lam_pc, 
        beta, 
        pre_trained_model,
        n_prun_c,
        training_mode,
        toml_file):

    config = get_paths(toml_file=toml_file)
    data_file = config['paths']['main_dir'] / config['paths']['data_path'] / config['data']['anndata_file']
    gene_file = config['paths']['main_dir'] / config['paths']['data_path'] / config['data']['hvg_file']
    
    folder_name = f'run_{n_run}_K_{n_categories}_Sdim_{state_dim}_p_drop_{p_drop}_fc_dim_{fc_dim}_aug_{augmentation}' + \
                  f'_lr_{lr}_n_arm_{n_arm}_tau_{tau}_nbatch_{batch_size}_nepoch_{n_epoch}_nepochP_{n_epoch_p}'
    
    saving_folder = config['paths']['main_dir'] / config['paths']['saving_path']
    saving_folder = saving_folder / folder_name
    os.makedirs(saving_folder, exist_ok=True)
    os.makedirs(saving_folder / 'model', exist_ok=True)
    saving_folder = str(saving_folder)

    if augmentation:
        aug_file = config['paths']['main_dir'] / config['paths']['models'] / config['models']['augmenter']
    else:
        aug_file = ''


    data = load_data(file=data_file, gene_file=gene_file, n_gene=n_gene) 

    mixvae = cpl_mixVAE(saving_folder=saving_folder, aug_file=aug_file, device=device)
    
    _, train_loader, test_loader, _, _, _ = get_loaders(
                                                        x=data['log1p'],
                                                        batch_size=batch_size, 
                                                        n_aug_smp=n_aug_smp, 
                                                        additional_val=False,
                                                        )

    mixvae.init_model(
                    n_categories=n_categories,
                    state_dim=state_dim,
                    input_dim=data['log1p'].shape[1],
                    fc_dim=fc_dim,
                    lowD_dim=latent_dim,
                    x_drop=p_drop,
                    s_drop=s_drop,
                    lr=lr,
                    n_arm=n_arm,
                    temp=temp,
                    hard=hard,
                    variational=variational,
                    tau=tau,
                    lam=lam,
                    lam_pc=lam_pc,
                    beta=beta,
                    ref_prior=ref_pc,
                    mode=training_mode,
                    trained_model=pre_trained_model,
                    n_pr=n_prun_c,
                    )
    

    mixvae.train(
                train_loader=train_loader,
                test_loader=test_loader,
                validation_loader=test_loader,
                n_epoch=n_epoch,
                n_epoch_p=n_epoch_p,
                min_con=min_con,
                max_prun_it=max_prun_it,
                )
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
