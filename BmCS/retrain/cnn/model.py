from .f_scores import F1Score
from os import mkdir
import os.path as os_path
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback, CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, TerminateOnNaN
from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv1D, Dense, Dropout, Embedding, Flatten, Input, Layer, MaxPooling1D
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


FSCORE_METRIC_NAME = 'fscore'


class Model:

    def build(self, model_config, pretrained_word_embeddings=None):

        word_embedding_layer = self._word_embedding_layer(model_config, pretrained_word_embeddings)
        conv_layers = self._create_conv_layers(model_config)

        title_input = Input(shape=(model_config.title_max_words,), name='title_input')
        title_word_embeddings = word_embedding_layer(title_input)
        title_features = self._text_feature_extraction(model_config, conv_layers, title_word_embeddings, 1)
        
        abstract_input = Input(shape=(model_config.abstract_max_words,), name='abstract_input')
        abstract_word_embeddings = word_embedding_layer(abstract_input)
        abstract_features = self._text_feature_extraction(model_config, conv_layers, abstract_word_embeddings, model_config.num_pool_regions)
      
        pub_year_input, pub_year = self._create_time_period_input(model_config.num_pub_year_time_periods, model_config.inputs_dropout_rate, 'pub_year_input')
        year_completed_input, year_completed = self._create_time_period_input(model_config.num_year_completed_time_periods, model_config.inputs_dropout_rate, 'year_completed_input')

        #journal_input, journal_embedding = self._journal_embedding(model_config.num_journals, model_config.journal_embedding_size, model_config.inputs_dropout_rate)
        
        hidden = Concatenate()([title_features, abstract_features, pub_year, year_completed])
        for layer_size in model_config.hidden_layer_sizes:
            hidden = Dense(layer_size, activation=None, use_bias=False)(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = Activation(model_config.hidden_layer_act)(hidden)
            hidden = Dropout(model_config.dropout_rate)(hidden)

        output = Dense(model_config.output_layer_size, activation=model_config.output_layer_act)(hidden)

        model = tensorflow.keras.models.Model(inputs=[title_input, abstract_input, pub_year_input, year_completed_input], outputs=[output])

        loss, optimizer, metrics, fscore_threshold = self._get_compile_inputs(model_config.init_learning_rate, model_config.init_threshold) 
        self._fscore_threshold = fscore_threshold
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        self._model = model

    def _create_conv_layers(self, model_config):
        conv_layers = []
        for filter_size in model_config.conv_filter_sizes:
            conv_layer = Conv1D(model_config.conv_num_filters, filter_size, activation=None, padding='valid', strides=1, use_bias=False)
            conv_layers.append(conv_layer)
        return conv_layers

    def _create_time_period_input(self, num_time_periods, dropout_rate, name):
        time_period_input = Input(shape=(num_time_periods,), name=name)
        time_period = Dropout(dropout_rate)(time_period_input)
        return time_period_input, time_period

    def _journal_embedding(self, num_journals, journal_embedding_size, dropout_rate):
        journal_input = Input(shape=(1,), name='journal_input')
        journal_embedding = Embedding(num_journals, journal_embedding_size, trainable=True)(journal_input)
        journal_embedding = Flatten()(journal_embedding)
        journal_embedding = Dropout(dropout_rate)(journal_embedding)
        return journal_input, journal_embedding

    def _text_feature_extraction(self, model_config, conv_layers, word_embeddings, num_pool_regions):
        conv_blocks = []
        for conv_layer in conv_layers:
            conv = conv_layer(word_embeddings)
            conv = BatchNormalization()(conv)
            conv = Activation(model_config.conv_act)(conv)
            pool_size = K.int_shape(conv)[1] // num_pool_regions
            conv = MaxPooling1D(pool_size=pool_size, strides=pool_size, padding='valid')(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)

        concat = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        text_features = Dropout(model_config.dropout_rate)(concat)
        return text_features

    def _word_embedding_layer(self, model_config, pretrained_word_embeddings):
        use_pretrained_word_embeddings = pretrained_word_embeddings is not None
        if use_pretrained_word_embeddings:
            vocab_size = pretrained_word_embeddings.shape[0]
            word_embedding_size = pretrained_word_embeddings.shape[1]
        else:
            vocab_size = model_config.vocab_size
            word_embedding_size = model_config.word_embedding_size
        if use_pretrained_word_embeddings:
            word_embedding_layer = EmbeddingWithDropout(model_config.word_embedding_dropout_rate, vocab_size, word_embedding_size, trainable=True, weights=[pretrained_word_embeddings])
        else:
            word_embedding_layer = EmbeddingWithDropout(model_config.word_embedding_dropout_rate, vocab_size, word_embedding_size, trainable=True)
        return word_embedding_layer

    def fit(self, root_config, training_data, dev_data, opt_data, output_dir):
        train_config = root_config.train
        resume_config = train_config.resume

        if resume_config.enabled:
            K.set_value(self._model.optimizer.lr, resume_config.learning_rate)
        
        # Ensure output dir exists
        self._mkdir(output_dir)
        
        # Save config
        settings_filepath = os_path.join(output_dir, train_config.save_config.settings_filename)
        model_json_filepath = os_path.join(output_dir, train_config.save_config.model_json_filename)
        model_img_filepath = os_path.join(output_dir, train_config.save_config.model_img_filename)
        with open(settings_filepath, 'wt', encoding=train_config.save_config.encoding) as settings_file:
            settings_file.write(str(root_config))
        with open(model_json_filepath, 'wt', encoding=train_config.save_config.encoding) as model_json_file:
            model_json_file.write(self._model.to_json(indent=4))
        plot_model(self._model, model_img_filepath, show_shapes=True, show_layer_names=True, rankdir='TB')

        callbacks = []

        # Optimize fscore threshold
        if train_config.optimize_fscore_threshold.enabled:
            optimize_fscore_threshold_callback = OptimizeFscoreThresholdCallback(self._fscore_threshold, train_config.optimize_fscore_threshold, opt_data)
            callbacks.append(optimize_fscore_threshold_callback)

        # Save checkpoints
        checkpoint_config = root_config.model.checkpoint
        if checkpoint_config.enabled:
            checkpoint_dir = os_path.join(output_dir, checkpoint_config.dir)
            self._mkdir(checkpoint_dir)
            checkpoint_filename = resume_config.resume_checkpoint_filename if resume_config.enabled else checkpoint_config.filename
            checkpoint_path = os_path.join(checkpoint_dir, checkpoint_filename)
            checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor=train_config.monitor_metric, verbose=0, save_best_only=True, mode=train_config.monitor_mode, save_weights_only=checkpoint_config.weights_only)
            callbacks.append(checkpoint_callback)

        # Terminate on NaN
        callbacks.append(TerminateOnNaN())

        # Reduce learning rate
        callbacks.append(ReduceLROnPlateau(train_config.monitor_metric, factor=train_config.reduce_learning_rate.factor, patience=train_config.reduce_learning_rate.patience, verbose=0, mode=train_config.monitor_mode, 
                                            min_delta=train_config.reduce_learning_rate.min_delta, cooldown=0, min_lr=0))

        # Early stopping
        callbacks.append(EarlyStopping(train_config.monitor_metric, min_delta=train_config.early_stopping.min_delta, patience=train_config.early_stopping.patience, verbose=0, mode=train_config.monitor_mode))
        
        # CSV logger
        csv_dir = os_path.join(output_dir, train_config.csv_logger.dir)
        self._mkdir(csv_dir)
        csv_filename = resume_config.resume_logger_filename if resume_config.enabled else train_config.csv_logger.filename
        csv_path = os_path.join(csv_dir, csv_filename)
        callbacks.append(CSVLogger(filename=csv_path))

        # Tensorboard
        if train_config.tensorboard.enabled:
            log_dir = os_path.join(output_dir, train_config.tensorboard.dir)
            self._mkdir(log_dir)
            tensorboard_callback = TensorBoard(log_dir, write_graph=train_config.tensorboard.write_graph)
            callbacks.append(tensorboard_callback)

        history = self._model.fit(training_data, epochs=train_config.max_epochs, verbose=1, callbacks=callbacks, 
                                 validation_data=dev_data, shuffle=True, initial_epoch=train_config.initial_epoch, use_multiprocessing=train_config.use_multiprocessing, workers=train_config.workers, max_queue_size=train_config.max_queue_size)
        logs = history.history

        monitor_mode_func = globals()['__builtins__'][train_config.monitor_mode]
        monitor_metric_values = logs[train_config.monitor_metric]
        best_epoch_index = monitor_metric_values.index(monitor_mode_func(monitor_metric_values))
        best_epoch_logs = { key: logs[key][best_epoch_index] for key in logs.keys() }
        best_epoch_logs['best epoch'] = best_epoch_index 

        best_epoch_logs_txt = self._str_format_logs(best_epoch_logs)
        print(best_epoch_logs_txt)

        best_epoch_logs_filepath = os_path.join(output_dir, train_config.csv_logger.dir, train_config.csv_logger.best_epoch_filename)
        with open(best_epoch_logs_filepath, 'wt', encoding=train_config.csv_logger.encoding) as best_epoch_logs_file:
            best_epoch_logs_file.write(best_epoch_logs_txt)

        return best_epoch_logs

    def restore(self, restore_config, input_dir):
        checkpoint_path = os_path.join(input_dir, restore_config.model_checkpoint_dir, restore_config.model_checkpoint_filename)
        loss, optimizer, metrics, _ = self._get_compile_inputs(restore_config.learning_rate, restore_config.threshold) 
        custom_objects = { EmbeddingWithDropout.__name__: EmbeddingWithDropout }
        if restore_config.weights_only_checkpoint:
            model_json_filepath = os_path.join(input_dir, restore_config.model_json_filename)
            with open(model_json_filepath, 'rt', encoding=restore_config.encoding) as model_json_file:
                model_json = model_json_file.read()
            self._model = tensorflow.keras.models.model_from_json(model_json, custom_objects=custom_objects)
            self._model.load_weights(checkpoint_path)
            self._model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        else:
            for metric in metrics:
                custom_objects[metric.__name__] = metric
            custom_objects[loss.__name__] = loss
            self._model = tensorflow.keras.models.load_model(checkpoint_path, custom_objects=custom_objects)

    def predict(self, pred_config, test_data):
        predictions = {}
        num_batches = len(test_data)
        for batch_index in range(num_batches):
            x = test_data[batch_index][0]
            y = test_data[batch_index][1]
            batch_scores = self._model.predict_on_batch(x)
            batch_size = len(batch_scores)
            for index in range(batch_size):
                pmid = int(x['pmids'][index])
                score = batch_scores[index][0]
                act = y[index][0]
                predictions[pmid] = { 'act': act, 'score': score}
        return predictions

    def _get_compile_inputs(self, learning_rate, threshold):
        loss = binary_crossentropy
        optimizer = Adam(lr=learning_rate)
        fscore_metric = F1Score(None, 'micro', threshold, name=FSCORE_METRIC_NAME)
        metrics = [fscore_metric]
        return loss, optimizer, metrics, fscore_metric.threshold 

    def _mkdir(self, dir):
        if not os_path.isdir(dir): 
            mkdir(dir) 

    def _str_format_logs(self, logs):
        txt = ' '.join(['{}: {:.9f}'.format(name, value) for name, value in sorted(logs.items(), key=lambda x:x[0])])
        return txt


class OptimizeFscoreThresholdCallback(Callback):

    def __init__(self, threshold, config, opt_data):
        super().__init__()
        self.config = config
        self.opt_data = opt_data
        self.threshold = threshold
        self.metric_index = -1

    def set_model(self, model):
        super().set_model(model)
        
    def on_batch_end(self, batch, logs = None):

        if self.metric_index < 0:
            for idx, metric_name in enumerate(self.model.metrics_names):
                if metric_name == self.config.metric_name:
                    self.metric_index = idx

        step = batch + 1
        if step == self.params['steps']: # Have finished the last batch
            self.prev_threshold_value = K.get_value(self.threshold)
            alpha = self.config.alpha
            k = self.config.k
            candidate_thresholds = [(self.prev_threshold_value - (alpha*k)) + (x*alpha) for x in range(2*k + 1)]
            best_metric_value = 0
            best_threshold = self.prev_threshold_value
            for candidate_threshold in candidate_thresholds:
                K.set_value(self.threshold, candidate_threshold)
                metrics = self.model.evaluate(self.opt_data, use_multiprocessing=self.config.use_multiprocessing, workers=self.config.workers, max_queue_size=self.config.max_queue_size)
                metric_value = metrics[self.metric_index]
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_threshold = candidate_threshold
            K.set_value(self.threshold, best_threshold)
            

    def on_epoch_end(self, epoch, logs = None):
        logs = logs or {}
        logs['threshold'] = self.prev_threshold_value
        logs['val_threshold'] = K.get_value(self.threshold)


class EmbeddingWithDropout(Embedding):

    def __init__(self, dropout_rate, *args, **kwargs):
        self.dropout_rate = dropout_rate
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        _embeddings = K.in_train_phase(K.dropout(self.embeddings, self.dropout_rate, noise_shape=[self.input_dim,1]), self.embeddings) if self.dropout_rate > 0 else self.embeddings
        out = K.gather(_embeddings, inputs)
        return out

    def get_config(self):
        config = { 'dropout_rate': self.dropout_rate }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))