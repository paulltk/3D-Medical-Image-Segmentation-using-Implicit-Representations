import numpy as np
import torch 
import matplotlib.pyplot as plt


def plot_graph(path, x, ys_and_labels, axes=("Epochs", "BCELoss"), fig_name="loss_plot"):
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    
    for y, label in ys_and_labels: 
        ax.plot(x[1:], y[1:], label=label)

    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    legend = ax.legend(loc='upper right')
    
    plt.savefig(f"{path}/{fig_name}.png")
    plt.close()

    
def save_loss(path, loss, models, optims, name="loss", save_models=True):
    np.save(f"{path}/{name}.npy", loss)
    
    eps, t_loss, v_loss = loss[:, 0], loss[:, 1], loss[:, 3]
    
    print(f"{name.ljust(15)} Train: {str(round(t_loss[-1], 6)).ljust(8, '0')}, \t Eval: {str(round(v_loss[-1], 6)).ljust(8, '0')}")
    
    if save_models:
        if t_loss[-1] == t_loss.min(): 
            print(f"New best train loss, saving model.")
            if save_models:
                for model in models.keys():
                    torch.save(models[model].state_dict(), f"{path}/{model}_{name}_train.pt")
                    torch.save(optims[model].state_dict(), f"{path}/{model}_optim_{name}_train.pt")
    
        if v_loss[-1] == v_loss.min(): 
            print(f"New best eval  loss, saving model.")
            if save_models:
                for model in models.keys():
                    torch.save(models[model].state_dict(), f"{path}/{model}_{name}_val.pt")
                    torch.save(optims[model].state_dict(), f"{path}/{model}_optim_{name}_val.pt")
        
    plot_graph(path, eps, [(t_loss, "Train loss"), (v_loss, "Eval loss")], 
               axes=("Epochs", "Loss"), fig_name=f"{name}_plot")