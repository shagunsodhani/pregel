import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from app.utils.constant import FF, GCN, GCN_AE, GCN_POLY, GCN_VAE

plt.switch_backend('agg')

def plot_loss_curves(train_loss_runs, validation_loss_runs, dataset_name, model_params):
    fontsize = 20
    def _tsplot(list_data, label, color):
        '''Wrapper method over tsplot'''
        data = np.asarray(list_data)
        y_axis = np.linspace(0, data.shape[1] - 1, data.shape[1])
        ax = sns.tsplot(data=data,
                        ci="sd",
                        color=color,
                        condition=[label],
                        legend=True,
                        time=y_axis,
                        )
        plt.setp(ax.get_legend().get_texts(), fontsize=fontsize)

        return ax

    train_ax = _tsplot(list_data=train_loss_runs,
                       color="b",
                       label="Loss for training data"
                       )

    val_ax = _tsplot(list_data=validation_loss_runs,
                     color="r",
                     label="Loss for validation data"
                     )
    val_ax.set(xlabel="Number of epochs", ylabel="Loss value")



    title = "Training and validation curve for {} dataset using {} model".format(
        dataset_name.capitalize(),
        model_params.model_name.upper())
    if(model_params.model_name in set([GCN, GCN_POLY, GCN_AE, GCN_VAE])):
        title = title+" with support size = " + str(model_params.support_size)

    for item in ([val_ax.xaxis.label, val_ax.yaxis.label] +
                     val_ax.get_xticklabels() +
                     val_ax.get_yticklabels()):
        item.set_fontsize(fontsize)

    plt.savefig(title + ".png", bbox_inches='tight')

    plt.show()


def print_stats(train_loss_runs, validation_loss_runs, test_metrics, test_metrics_labels):
    '''Method to print the stats after training'''
    train_loss = np.asarray(train_loss_runs)
    validation_loss = np.asarray(validation_loss_runs)

    print("Training loss after {} epochs, averaged over {} runs = {}".format(
        train_loss.shape[1],
        train_loss.shape[0],
        np.average(train_loss[:, -1])
    ))

    print("Validation loss after {} epochs, averaged over {} runs = {}".format(
        validation_loss.shape[1],
        validation_loss.shape[0],
        np.average(validation_loss[:, -1])
    ))

    for metric, label in zip(test_metrics, test_metrics_labels):
        best_test_metric = []
        for i in range(validation_loss.shape[0]):
            best_test_metric.append(
                np.max(metric[i])
            )
        best_test_metric = np.asarray(best_test_metric)

        print("{} over test data after fitting, averaged over {} runs = {}".format(
            label,
            best_test_metric.shape[0],
            np.average(best_test_metric)
        ))

def embedd_and_plot(node_representation, labels, mask):
    '''Method to compute and plot the t_sne embeddings for given node representation'''
    node_embedding = compute_embeddings(node_representation)
    if(len(labels.shape)==2):
    #     k-hot label provided
        labels = np.argmax(labels, axis=1)
    labels=labels[mask>0]
    node_embedding = node_embedding[mask>0]
    plt.scatter(node_embedding[:,0], node_embedding[:,1], c = labels)
    # plt.show()


def compute_embeddings(node_representation):
    '''Method to compute the t_sne embeddings for given node representation'''
    return TSNE(n_components = 2).fit_transform(node_representation)
