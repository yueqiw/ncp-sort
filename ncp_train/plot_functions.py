

import numpy as np
import matplotlib.pyplot as plt

def plot_avgs(losses, accs, rot_vars, w, save_name=None):
    
    up = -1 #3500
    
    avg_loss = []
    for i in range(w, len(losses)):
        avg_loss.append(np.mean(losses[i-w:i]))
    
    avg_acc = []
    for i in range(w, len(accs)):
        avg_acc.append(np.mean(accs[i-w:i]))
    
    avg_var = []
    for i in range(w, len(rot_vars)):
        avg_var.append(np.mean(rot_vars[i-w:i]))
    
    
    plt.figure(22, figsize=(13,10))
    plt.clf()
    
    plt.subplot(312)
    plt.semilogy(avg_loss[:up])
    plt.ylabel('Mean NLL')
    plt.grid()
    
    plt.subplot(311)
    plt.plot(avg_acc[:up])
    plt.ylabel('Mean Accuracy')
    plt.grid()
    
    plt.subplot(313)
    plt.semilogy(avg_var[:up])
    plt.ylabel('NLL std/mean')
    plt.xlabel('Iteration')
    plt.grid()

    if save_name:
        plt.savefig(save_name)
        plt.close()
















        


        
