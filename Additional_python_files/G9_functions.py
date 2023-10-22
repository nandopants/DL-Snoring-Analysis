#This python file contains functions relevant to the G9 snoring project


from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve,roc_curve, roc_auc_score
import pandas as pd
#import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


#---------------------------------------------------------------------------
#precision recall vs threshold plot
#if crossover is true it will show the threshold for the precsion recall crossover.
#----------------------------------------------------------------------
def precision_recall_threshold_charts(y_true,y_proba,crossover=False):

    #obtain precision, recall and threshold values.
    precisions, recalls, thresholds = precision_recall_curve(y_true,y_proba)
    
    #plot
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5))
    
    axs[0].set_title("Precision-Recall vs Threshold Chart")
    axs[0].plot(thresholds,precisions[:-1], "b--", label="Precision", linewidth=2) #precision 
    axs[0].plot(thresholds,recalls[:-1], "g-", label="Recall", linewidth=2) #recall
    axs[0].set_xlabel("Threshold")
    axs[0].set_ylim([0,1])
    if crossover is True:
        # Find the threshold where precision and recall intersect
        crossover_index = find_crossover_index(precisions,recalls)
        if crossover_index is False:
            return print("There is no crossover point.")
        
        crossover_threshold = thresholds[crossover_index]
        axs[0].plot(crossover_threshold, precisions[crossover_index], 'ro')  # Mark the crossover point
        axs[0].annotate(f'Crossover= {crossover_threshold:.2f}',
                     xy=(crossover_threshold, precisions[crossover_index]),
                     xytext=(crossover_threshold, precisions[crossover_index] - 0.1),
                     arrowprops=dict(facecolor='black', arrowstyle='wedge,tail_width=0.7', alpha=0.5),
                     fontsize=12,
                     ha='center')
    axs[0].grid(True)

    axs[0].legend(loc="lower left")
    #precison vs recall plot on right
    axs[1].set_title("Precision-Recall Curve")
    axs[1].plot(recalls,precisions, "b--", label="Precision/Recall Curve", linewidth=2) #recall vs precision line
    axs[1].plot(recalls[crossover_index], precisions[crossover_index], "ro",label=f'Crossover Threshold = {crossover_threshold:.2f}') # crossover threshold value
    axs[1].vlines(recalls[crossover_index],0,precisions[crossover_index],"k","dotted")
    axs[1].hlines(precisions[crossover_index],0,recalls[crossover_index],"k","dotted")
    axs[1].set_xlabel("Recall", fontsize=16)
    axs[1].set_ylabel("Precision", fontsize=16)
    axs[1].axis([0, 1.02, 0, 1.02])
    axs[1].legend(loc="lower left")
    axs[1].grid(True)
    
    plt.show()


#------------------------------------
#function to find the threshold value when precision and recall crossover.
#-----------------------------------
def find_crossover_index(precisions, recalls):
    for i, (p, r) in enumerate(zip(precisions, recalls)):
        if p >= r:
            return i
    return False

#---------------------------------------
#Function to see the ROC curve of single ot multiple models
#must enter the predicted values as a dictionary
#---------------------------------------

def plot_roc_curves(y_true, y_pred_dic):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC curves")

    for label, y_pred in y_pred_dic.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_pred)
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC={auc_score:.3f}")

    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Random Classifier")
    ax.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    
#----------------------------------------------
# Function to plot the learning curves for tensorflow model history
# Model history must be entered as a dictionary
# Is designed to have multiple or a single model to compare.
#-------------------------------------------------
def plot_learning_curves(history_dict):

    initial_colors = ['r', 'b', 'g', 'c', 'm', 'y'] 
    colors = initial_colors + [f'C{i}' for i in range(len(initial_colors), 20)]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5))
    
    if len(history_dict) == 1:
        first_item = next(iter(history_dict.items()))
        first_item[1].history
        
        axs[0].plot(pd.DataFrame( first_item[1].history)[["loss"]], "--", color='r', label='Train Loss')
        axs[0].plot(pd.DataFrame( first_item[1].history)[["val_loss"]], "-", color='b', label='Val Loss')
        axs[1].plot(pd.DataFrame( first_item[1].history)[["accuracy"]], "--", color='r', label='Training Accuracy')
        axs[1].plot(pd.DataFrame( first_item[1].history)[["val_accuracy"]], "-", color='b', label='Val Accuracy')
        
    
    else:
        for idx, (label, history) in enumerate(history_dict.items()):
            color = colors[idx % len(colors)]  # Cycle through colors if more than len(colors) models
            axs[0].plot(pd.DataFrame(history.history)[["loss"]], "--", color=color, label=f'Train Loss {label}')
            axs[0].plot(pd.DataFrame(history.history)[["val_loss"]], "-", color=color, label=f'Val Loss {label}')
            axs[1].plot(pd.DataFrame(history.history)[["accuracy"]], "--", color=color, label=f'Training Accuracy {label}')
            axs[1].plot(pd.DataFrame(history.history)[["val_accuracy"]], "-", color=color, label=f'Val Accuracy {label}')

    axs[0].set_title('Training Loss and Validation Loss Over Epochs')
    axs[0].set_xlabel('Epochs', fontsize=10)
    axs[0].set_ylabel('Training and Val Loss', fontsize=10)
    axs[0].legend()
    axs[0].grid(True, which='both')

    axs[1].set_title('Train Accuracy and Val Accuracy Over Epochs')
    axs[1].set_xlabel('Epochs', fontsize=10)
    axs[1].set_ylabel('Training and Val Accuracy', fontsize=10)
    axs[1].yaxis.set_major_locator(plt.MultipleLocator(base=0.02))
    axs[1].tick_params(axis='y', labelsize=8)
    axs[1].tick_params(axis='x', labelsize=8)
    axs[1].legend()
    axs[1].grid(True, linestyle='-', linewidth=1)

    plt.show()
    
