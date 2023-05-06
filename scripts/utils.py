'''
utils

各種共通関数
'''
################################
# Libraries
################################
import os
from datetime import datetime as dt
import matplotlib.pyplot as plt
# evaluation
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc

################################
# Functions
################################

# Prepare directory
def prepare_savedir(ROOT:str, name:str)->str:
    '''
    Args
        ROOT (str) : Directory for save
        name (str) : Save directory name
    '''

    # define directory
    savedir = os.path.join(ROOT, name)
    # make direcotry
    os.makedirs(savedir, exist_ok=True)

    return savedir

def getdate():
    '''
    Args
        None
    '''

    return dt.now().strftime("%Y%m%d_%H%M")


# matplot graph setting
def graph_setting():
    '''
    Args
        None
    '''
    # setting
    plt.rcParams["figure.dpi"] = 100 # dpi(dots per inch, resolution )

    # font setting
    plt.rcParams["font.serif"] = "Times New Roman"
    plt.rcParams["font.size"] = 12

    # xticls setting
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["xtick.major.size"] = 4
    plt.rcParams["xtick.major.width"] = 1
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["xtick.color"] = "black"
    # minor xticks setting
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["xtick.minor.size"] = 2.0
    plt.rcParams["xtick.minor.width"] = 0.6

    # yticks setting
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["ytick.major.size"] = 4
    plt.rcParams["ytick.major.width"] = 1.0
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["ytick.color"] = "black"
    # minor xticks setting
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["ytick.minor.size"] = 2.0
    plt.rcParams["ytick.minor.width"] = 0.6

    # axis
    plt.rcParams["axes.labelsize"] = 12 # axes label font size
    plt.rcParams["axes.linewidth"] = 1.0 # outline width
    plt.rcParams["axes.grid"] = True # grid, true or false
    plt.rcParams["axes.edgecolor"] = "black"

    # grid
    plt.rcParams["grid.color"] = "black"
    plt.rcParams["grid.linewidth"] = 0.1

    # face color
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    # legend
    plt.rcParams["legend.loc"] = "best" # position of legend
    plt.rcParams["legend.frameon"] = True # frame of legend
    plt.rcParams["legend.framealpha"] = 1.0 # transparency of frame
    plt.rcParams["legend.facecolor"] = "white" # back color of legend
    plt.rcParams["legend.edgecolor"] = "black" # edge color
    plt.rcParams["legend.fancybox"] = True # edge type

# 学習曲線
def draw_learning_curve(estimator, X_train, y_train):
    # learning curve
    train_sizes, train_scores, test_scores = learning_curve(estimator=estimator, X=X_train, y=y_train, train_sizes=np.linspace(0.1,1,10), cv=10, n_jobs=1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # plot
    plt.figure(figsize=(10,6))
    # train data
    plt.plot(train_sizes, train_mean, color="blue", marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean+train_std, train_mean-train_std, color="blue", alpha=0.15)
    # val data
    plt.plot(train_sizes, test_mean, color="green", marker='s', linestyle='--', markersize=5, label='validation accuracy')
    plt.fill_between(train_sizes, test_mean+test_std, test_mean-test_std, color="green", alpha=0.15)

    plt.xlabel("Number of trainig samples")
    plt.ylabel("Accuracy")
    plt.ylim([0.2,1.0])
    plt.title("Learning curve")
    plt.legend()

# 混同行列と各種スコア
def confmat_roccurve(X_test, y_test, estimator):
    # prediction
    y_pred = estimator.predict(X_test)
    # create confusion matrix
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    # visualiazation confusion matrix
    fig, ax = plt.subplots(1,2,figsize=(18,6))

    ax[0].matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax[0].text(x=j, y=i, s=confmat[i,j], va="center", ha="center")

    ax[0].set_xlabel("predicted label")
    ax[0].set_ylabel("true label")
    ax[0].set_title("confusion matrix")
    # Score
    print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))
    print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))
    print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))
    print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))

    # visualization roc curve
    y_score = estimator.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score)
    ax[1].plot(fpr, tpr, label="roc curve (area = %.3f)" % auc(fpr, tpr), color="blue")
    ax[1].plot([0,1], [0,1], linestyle='--', color=(0.6,0.6,0.6), label='random')
    ax[1].plot([0,0,1], [0,1,1], linestyle=':', color="black", label='perfect performance')
    ax[1].set_xlabel("false positive rate")
    ax[1].set_ylabel("true positive rate")
    ax[1].set_title("Receiver Operator Characteristic")
    ax[1].legend()