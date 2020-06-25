import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

from os.path import join, splitext, isfile
from os import listdir, makedirs

from math import pow, floor
import re
import numpy as np
from sklearn.metrics import f1_score
from pandas import read_csv, DataFrame, Series
from time import time

from .plotting import plot_confusion_matrix

WEIGHT_FILENAME = re.compile("weights_v(?P<version>\d{1,3})")


class vgg19:
    def __init__(self, data_dir, weights_filename=None, input_shape=(224, 224, 3), class_names=None, **kwargs):
        """data_dir must be provided, which is the directory that contains the Train, Val, and Test directories to use
        in model training and testing functions. If weights_filename is given then a model is loaded otherwise a model
        is trained. Note that all files are relative to the data_dir provided."""
        self.data_dir = data_dir
        self.input_shape = input_shape
        self._build_model(**kwargs)

        # infer the class labels from the data directory unless explicitly provided in the class_names parameter (list)
        # - note that getting the class names from directories will always put them in alphabetical order.
        if class_names is None:
            self.class_names = sorted(listdir(join(data_dir, 'Train')))[:2]  # only binary is supported currently
        else:
            self.class_names = class_names

        # either load model or train a new model
        if weights_filename is not None:
            self.weights_filename = weights_filename
            self.model.load_weights(join(data_dir, weights_filename))
            print('weights loaded to model from {}'.format(weights_filename))
        else:
            t_start = time()
            self._train(**kwargs)
            delta_t = time() - t_start
            print('time taken to train: {:.0f} min and {:.0f}'.format(delta_t // 60, delta_t % 60))

    def _build_model(self, l2_lambda=0.004, **kwargs):
        """build the model
        modification of the model follows the example given in:
            https://github.com/geifmany/cifar-vgg/blob/master/cifar10vgg.py"""
        # include top false - the end of the model is modified in this implementation
        # - always start from imagenet weights
        model = VGG19(include_top=False, weights='imagenet', input_shape=self.input_shape)

        # allow weights in all layers to be adjusted during training
        # - this is the default behavior, we show to be transparent
        for layer in model.layers:
            layer.trainable = True

        # add dense layer of 512 nodes with L2 regularization and relu activation
        x = model.output
        x = Flatten()(x)
        x = Dense(512, kernel_regularizer=l2(l2_lambda), activation="relu")(x)
        # add batch normalization - VGG19 does not include batch normalization by default
        x = BatchNormalization()(x)
        # add 50% dropout
        x = Dropout(0.5)(x)
        # prediction layer is a sigmoid since it is assumed this is a binary problem
        # - future versions will allow handling of multiple classes
        predictions = Dense(1, activation='sigmoid')(x)
        self.model = Model(inputs=model.input, outputs=predictions)

    def _train(self, start_lr=0.001, lr_decay_rate=0.1, lr_decay_step=8.0, batch_size=8, shuffle_seed=64,
               maxepochs=100):
        """train the model using data in data_dir, which should contain the Train, Val, and Test directories with the
        two classes subdirs inside each. Learning rate decay is implemented during training and shuffle is used for
        the training dataset, which is seeded for reproducibility. Default parameters can be changed by passing kwargs
        to class init.

        start_lr - float, the starting learning rate
        lr_decay_rate - float, the % the learning rate will decay every decay step
        lr_decay_step - float, the number of epochs between each learning rate decay
        batch_size - int, set to small number (8 or 16) if computer can't handle default size
        shuffle_seed - int, random seed used when shuffling the training data
        """

        def _step_decay(epoch):
            """early decay internal function, source below
            https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-
            2c8f433990d1"""
            learning_rate = start_lr * pow(lr_decay_rate, floor((1 + epoch) / lr_decay_step))
            return learning_rate

        # callbacks - early stopping monitoring validation loss and learning reate decay by epochs
        early = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')  # min_delta=1
        lrate = LearningRateScheduler(_step_decay)

        # for model weights, check hdf5 files and save the name with appended int denoting version
        weight_filenames = [splitext(filename)[0] for filename in listdir(self.data_dir) if filename.endswith('.hdf5')]
        if len(weight_filenames):
            weight_filenames.sort()
            version = int(WEIGHT_FILENAME.search(weight_filenames[-1]).groupdict()['version']) + 1
        else:
            version = 1
        weights_filename = 'weights_v{}.hdf5'.format(version)
        self.weights_filename = weights_filename
        cp = ModelCheckpoint(join(self.data_dir, weights_filename), monitor='val_loss', verbose=1, save_best_only=True,
                             mode='min')

        # create data generators
        train_gen = ImageDataGenerator(rescale=1. / 255, rotation_range=270, vertical_flip=True,
                                       preprocessing_function=preprocess_input).flow_from_directory(
            join(self.data_dir, 'Train/'),
            target_size=self.input_shape[0:2],
            color_mode='rgb',
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True,
            seed=shuffle_seed,
            classes=self.class_names
        )

        val_gen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_input).flow_from_directory(
            join(self.data_dir, 'Val/'),
            target_size=self.input_shape[0:2],
            color_mode='rgb',
            shuffle=False,
            batch_size=batch_size,
            class_mode='binary',
            classes=self.class_names
        )

        # compile the model, using stochastic gradient decent
        self.model.compile(loss="binary_crossentropy", optimizer=SGD(lr=start_lr), metrics=["accuracy"])

        # train model with validation
        history = self.model.fit_generator(
            train_gen, epochs=maxepochs, steps_per_epoch=train_gen.n // batch_size,
            validation_data=val_gen, validation_steps=val_gen.n // batch_size,
            callbacks=[early, lrate, cp]
        )

        # predict on validation dataset to calculate F1-score, gives a better sense of model performance, use a default
        # - threshold of 0.5 for deciding labels on prediction
        pred_probs = self.model.predict(val_gen).ravel()
        pred_labels = (pred_probs > 0.5).astype(np.int)
        true_labels = val_gen.classes
        f1score = f1_score(true_labels, pred_labels)

        # create report DataFrame to save to CSV
        lr = history.history['lr']
        rows = {'weights_filename': weights_filename,
                'start_lr': start_lr, 'lr_decay_rate': lr_decay_rate, 'lr_decay_step': lr_decay_step,
                'batch_size': batch_size, 'maxepochs': maxepochs, 'shuffle_seed': shuffle_seed, 'epoch_stop': len(lr),
                'data_dir': self.data_dir, 'input_shape': str(self.input_shape),
                'class_names': ','.join(self.class_names), 'f1_score': float(f1score),
                'number_train_images': int(train_gen.n),
                'training_loss': ','.join(map(str, history.history['loss'])),
                'training_accuracy': ','.join(map(str, history.history['accuracy'])),
                'validation_loss': ','.join(map(str, history.history['val_loss'])),
                'validation_accuracy': ','.join(map(str, history.history['val_accuracy'])),
                'learning_rate': ','.join(map(str, lr))
                }

        # - check if report already exists, if so we read it and append another row of training results
        report_filepath = join(self.data_dir, 'report.csv')
        if isfile(report_filepath):
            df = read_csv(report_filepath)
        else:
            df = DataFrame(columns=list(rows.keys()))
        df = df.append(Series(rows), ignore_index=True)

        # save the report to csv file
        df.to_csv(report_filepath, index=False)
        print('done training, weights saved to filename {}'.format(weights_filename))

    def predict(self, dataset='Test', batch_size=16, threshold=0.5):
        """use trained model to predict on a dataset, by default this is the test dataset but it could be the Train and
        Val dataset if desired. Generates confusion matrix plot and saves it to file with F1-score in title. Also,
        creates a report of true / predicted labels for each image in the dataset and saves to CSV file."""
        gen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_input).flow_from_directory(
            join(self.data_dir, dataset),
            target_size=self.input_shape[0:2],
            color_mode='rgb',
            shuffle=False,
            batch_size=batch_size,
            class_mode='binary',
            classes=self.class_names
        )

        # compile the model, using stochastic gradient decent
        self.model.compile(loss="binary_crossentropy", optimizer=SGD(lr=0.001), metrics=["accuracy"])

        # predict on the dataset
        pred_probs = self.model.predict(gen).ravel()
        pred_labels = (pred_probs > threshold).astype(np.int)
        true_labels = gen.classes

        # calculate evaluation metrics
        f1score = f1_score(true_labels, pred_labels)

        # create directory to save figures
        figs_dir = join(self.data_dir, 'Figures')
        makedirs(figs_dir, exist_ok=True)

        # plot the confusion matrix, ROC and PRC and save to PNGs
        _ = plot_confusion_matrix(true_labels, pred_labels, labels=self.class_names,
                                  title='F1-score: {:.2f}'.format(f1score),
                                  save_path=join(figs_dir, '{}_{}_confusion_matrix.png'.format(
                                      splitext(self.weights_filename)[0], dataset)))

        # create a dataframe report for each image in dataset, with true labels and the predicted label
        rows = {'filepath': [], 'true_label': [], 'predicted_label': []}
        for filepath, pred, true in zip(gen.filepaths, pred_labels, true_labels):
            rows['filepath'].append(filepath)
            rows['true_label'].append(self.class_names[true])
            rows['predicted_label'].append(self.class_names[pred])
        df = DataFrame(data=rows)
        df.to_csv(join(self.data_dir, '{}_{}_image_report.csv'.format(splitext(self.weights_filename)[0], dataset)),
                  index=False)
