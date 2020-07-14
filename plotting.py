import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib
import datetime

cifar10classes = ["airplane", "automobile", "bird", "cat", "deer", "DOG", "frog", "horse", "ship", "truck"]

def save_images(k, loader, selected_conf_indices, confidences, logits, dset_name, model_folder, descr='pred', index_selection='highest', short_name=None):
            selected_conf_images = torch.zeros((k,) + loader.dataset[0][0].shape)
            selected_conf_labels = torch.zeros((k,))
            selected_confs = np.zeros((k,))
            selected_confs_decisions = np.zeros((k,))
            for i in range(k):
                selected_conf_images[i] = loader.dataset[selected_conf_indices[i]][0]
                selected_conf_labels[i] = loader.dataset[selected_conf_indices[i]][1]
                selected_confs[i] = confidences[selected_conf_indices[i]]
                selected_confs_decisions[i] = np.argmax(logits[selected_conf_indices[i]], axis=0)
            #save_path = model_folder + '/sample_images/{0}pred_out_selected_conf_images'.format(str(test_loader_out.dataset).split()[1])            
            #vutils.save_image(pred_out_selected_conf_images.detach(), save_path + '.png', normalize=False)
            
            selected_conf_images_np = selected_conf_images.detach().cpu().numpy()
            f,ax = plt.subplots(3, k, dpi=100, figsize=(k,3))
            normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
            for i in range(k):
                if selected_conf_images_np.shape[-3] == 1:
                    ax[1,i].imshow(selected_conf_images_np[i].squeeze(), cmap='gray', norm=normalize)
                else:
                    assert selected_conf_images_np.shape[-3] == 3
                    ax[1,i].imshow(np.transpose(selected_conf_images_np[i], (1,2,0)), norm=normalize)
                ax[1,i].axis('off')
                if dset_name == 'CIFAR10':
                    ax[0,i].text(0.2, 0.1, cifar10classes[int(selected_conf_labels[i].item())])
                else:
                    ax[0,i].text(0.2, 0.1, str(int(selected_conf_labels[i].item())))
                ax[0,i].axis('off')
                if dset_name == 'CIFAR10':
                    ax[2,i].text(0.2, 0.7, cifar10classes[int(selected_confs_decisions[i].item())])
                else:
                    ax[2,i].text(0.2, 0.7, str(int(selected_confs_decisions[i].item())))
                ax[2,i].text(0.2, 0.4, '{:.4f}'.format(selected_confs[i].item()))
                ax[2,i].axis('off')
            save_path = model_folder + f'/sample_images/{short_name}_{dset_name}_{descr}_{index_selection}_conf'
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
            file = open(save_path + '.txt', 'w+')
            file.write(f'{descr}_{dset_name}_{index_selection}_conf_indices: {0}\n')
            file.write(f'{descr}_{dset_name}_{index_selection}_conf_labels: {0}\n')
            file.write(f'{descr}_{dset_name}_{index_selection}_confs: {0}\n')
            file.write(f'{descr}_{dset_name}_{index_selection}_confs_decisions: {0}\n')
            file.close()
            
            plt.hist(confidences, bins=50, range=(0.1, 1.0), log=False, label= f'{dset_name}_{descr}_confidences')
            plt.ylim(ymin=0, ymax=len(confidences)/(1.5))
            plt.xlim(xmin=0.09, xmax=1.01)
            plt.axvline(confidences.mean(), color='k', linestyle='dashed', linewidth=2)
            save_path = model_folder + f'/sample_images/{short_name}_{dset_name}_{descr}_confidences_hist'
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()