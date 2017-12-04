import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_loss_curves(train_loss_runs, validation_loss_runs, dataset_name, model_name):
    def _tsplot(list_data, label, color):
        '''Wrapper method over tsplot'''
        data = np.asarray(list_data)
        y_axis = np.linspace(0, data.shape[1] - 1, data.shape[1])
        ax = sns.tsplot(data=data,
                        ci="sd",
                        marker="o",
                        color=color,
                        condition=[label],
                        legend=True,
                        time=y_axis
                        )
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
        dataset_name,
        model_name)
    plt.title(title)
    plt.savefig(title + ".png")
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
                metric[i][np.argmin(validation_loss[i, :])]
            )
        best_test_metric = np.asarray(best_test_metric)

        print("{} over test data after fitting, averaged over {} runs = {}".format(
            label,
            best_test_metric.shape[0],
            np.average(best_test_metric)
        ))
