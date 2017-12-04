import tensorflow as tf
from tensorflow.contrib.keras import backend as K

from app.app.util import plot_loss_curves, print_stats
from app.ds.data_pipeline import DataPipeline
from app.model.model_select import select_model
from app.model.params import SparseModelParams
from app.utils.constant import *


def run(model_params, data_dir, dataset_name):
    datapipeline = DataPipeline(model_params=model_params,
                                data_dir=data_dir,
                                dataset_name=dataset_name)

    placeholder_dict = datapipeline.get_placeholder_dict()

    feed_dict_train = datapipeline.get_feed_dict(mode=TRAIN)

    feed_dict_val = datapipeline.get_feed_dict(mode=VALIDATION)

    feed_dict_test = datapipeline.get_feed_dict(mode=TEST)

    sparse_model_params = SparseModelParams(
        num_elements=datapipeline.num_elements,
        feature_size=datapipeline.feature_size
    )

    sess = tf.Session()
    K.set_session(sess)

    train_loss_runs = []
    validation_loss_runs = []
    test_accuracy_runs = []

    for _ in range(model_params.num_exp):

        model = select_model(model_name=model_params.model_name)(
            model_params=model_params,
            sparse_model_params=sparse_model_params,
            placeholder_dict=placeholder_dict
        )

        if (model_params.tensorboard_logs_dir):
            train_writer = tf.summary.FileWriter(
                model_params.tensorboard_logs_dir + model_params.model_name + "/" + TRAIN,
                sess.graph)
            val_writer = tf.summary.FileWriter(
                model_params.tensorboard_logs_dir + model_params.model_name + "/" + VALIDATION, sess.graph)
        sess.run(tf.global_variables_initializer())

        train_loss_list = []
        validation_loss_list = []
        test_accuracy_list = []

        for epoch in range(model_params.epochs):
            loss, accuracy, opt, summary = sess.run([model.loss, model.accuracy, model.optimizer_op, model.summary_op],
                                                    feed_dict=feed_dict_train)

            loss_val, accuracy_val, summary_val = sess.run([model.loss, model.accuracy, model.summary_op],
                                                           feed_dict=feed_dict_val)

            if (model_params.tensorboard_logs_dir):
                train_writer.add_summary(summary, epoch)
                val_writer.add_summary(summary_val, epoch)

            train_loss_list.append(loss)
            validation_loss_list.append(loss_val)

            accuracy_test = sess.run([model.accuracy], feed_dict=feed_dict_test)
            test_accuracy_list.append(accuracy_test)

        train_loss_runs.append(train_loss_list)
        validation_loss_runs.append(validation_loss_list)
        test_accuracy_runs.append(test_accuracy_list)

    plot_loss_curves(train_loss_runs, validation_loss_runs, dataset_name=dataset_name,
                     model_name=model_params.model_name)
    print_stats(train_loss_runs, validation_loss_runs, test_metrics=[test_accuracy_runs],
                test_metrics_labels=[ACCURACY])
