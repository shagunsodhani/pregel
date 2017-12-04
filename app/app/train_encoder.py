import tensorflow as tf
from tensorflow.contrib.keras import backend as K

from app.ds.data_pipeline_ae import DataPipelineAE
from app.model.model_select import select_model
from app.utils.constant import *
from app.utils.metrics import compute_auc_score, compute_average_precision_recall
from app.app.util import plot_loss_curves, print_stats



def run(model_params, data_dir, dataset_name):
    datapipeline = DataPipelineAE(model_params=model_params,
                                data_dir=data_dir,
                                dataset_name=dataset_name)

    feed_dict_train = datapipeline.get_feed_dict(mode=TRAIN)

    feed_dict_val = datapipeline.get_feed_dict(mode=VALIDATION)

    feed_dict_test = datapipeline.get_feed_dict(mode=TEST)


    sess = tf.Session()
    K.set_session(sess)

    train_loss_runs = []
    validation_loss_runs = []
    test_aucscore_runs = []
    test_apr_runs = []

    for _ in range(model_params.num_exp):

        model = select_model(model_name=model_params.model_name)(
            model_params=model_params,
            sparse_model_params=datapipeline.get_sparse_model_params(),
            placeholder_dict=datapipeline.get_placeholder_dict(),
            autoencoder_model_params=datapipeline.get_autoencoder_model_params()
        )

        if (model_params.tensorboard_logs_dir):
            train_writer = tf.summary.FileWriter(
                model_params.tensorboard_logs_dir + model_params.model_name + "/" + TRAIN,
                sess.graph)
            val_writer = tf.summary.FileWriter(
                model_params.tensorboard_logs_dir + model_params.model_name + "/" + VALIDATION, sess.graph)

        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()])

        train_loss_list = []
        validation_loss_list = []
        test_aucscore_list = []
        test_apr_list = []
        for epoch in range(model_params.epochs):
            loss, accuracy, opt, summary = sess.run([model.loss, model.accuracy, model.optimizer_op, model.summary_op],
                                                    feed_dict=feed_dict_train)

            loss_val, accuracy_val, summary_val = sess.run(
                [model.loss,
                 model.accuracy,
                 model.summary_op],
                feed_dict=feed_dict_val)

            if (model_params.tensorboard_logs_dir):
                train_writer.add_summary(summary, epoch)
                val_writer.add_summary(summary_val, epoch)

            embedding, predictions_test, labels_test, mask_test, loss_test, accuracy_test, summary_test = sess.run(
                [model.embeddings,
                 model.logits,
                 model.labels,
                 model.mask,
                 model.loss,
                 model.accuracy,
                 model.summary_op],
                feed_dict=feed_dict_test)

            test_aucscore_list.append(compute_auc_score(labels=labels_test,
                                                        predictions=predictions_test,
                                                        mask=mask_test))

            test_apr_list.append(compute_average_precision_recall(labels=labels_test,
                                                        predictions=predictions_test,
                                                        mask=mask_test))

            train_loss_list.append(loss)
            validation_loss_list.append(loss_val)

        train_loss_runs.append(train_loss_list)
        validation_loss_runs.append(validation_loss_list)
        test_aucscore_runs.append(test_aucscore_list)
        test_apr_runs.append(test_apr_list)

    # plot_loss_curves(train_loss_runs, validation_loss_runs, dataset_name=dataset_name,
    #                  model_name=model_params.model_name)
    print_stats(train_loss_runs, validation_loss_runs, test_metrics=[test_aucscore_runs, test_apr_runs],
                test_metrics_labels=[AUCSCORE, AVERAGE_PRECISION_RECALL_SCORE])
