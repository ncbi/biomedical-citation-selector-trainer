import json
import os.path as os_path


ENCODING = 'utf8'


def get_config(data_dir=None, runs_dir=None):
    machine_config = _MachineConfig()
    if data_dir:
        machine_config.data_dir = data_dir
    if runs_dir:
        machine_config.runs_dir = runs_dir
    config = Config(machine_config)
    return config


class _ConfigBase:
    def __init__(self, parent, machine_config):
        self._parent = parent
        self._initialize(machine_config)

    def _initialize(self, machine_config):
        pass

    def __str__(self):
        dict = {}
        self._toJson(dict, self)
        return json.dumps(dict, indent=4)
            
    @classmethod
    def _toJson(cls, parent, obj):
        for attribute_name in dir(obj): 
            if not attribute_name.startswith('_'):
                attribute = getattr(obj, attribute_name)
                if isinstance(attribute, _ConfigBase):
                    child = {}
                    parent[attribute_name] = child
                    cls._toJson(child, attribute)
                else:
                    parent[attribute_name] = attribute


class _MachineConfig:
    def __init__(self):
     
        self.data_dir = None
        self.runs_dir = None
        self.use_multiprocessing = False
        self.workers = 1      
        self.max_queue_size = 1
        

class _CheckpointConfig(_ConfigBase):
    def _initialize(self, _):
  
        self.enabled = True
        self.weights_only = True
        self.dir = 'checkpoints'
        self.filename = 'best_model.hdf5'


class _CsvLoggerConfig(_ConfigBase):
    def _initialize(self, _):

        self.dir = 'logs'
        self.filename  = 'logs.csv'
        self.best_epoch_filename = 'best_epoch_logs.txt'
        self.encoding = ENCODING


class _EarlyStoppingConfig(_ConfigBase):
    def _initialize(self, _):

        self.min_delta = 0.001
        self.patience = 2


class _ModelConfig(_ConfigBase):
    def _initialize(self, machine_config):

        self.checkpoint = _CheckpointConfig(self, machine_config)

        self.word_embedding_size = 300
        self.word_embedding_dropout_rate = 0.25
     
        self.conv_act = 'relu'
        self.num_conv_filter_sizes = 3
        self.min_conv_filter_size = 2
        self.conv_filter_size_step = 3
        self.total_conv_filters = 350
        self.num_pool_regions = 5

        self.num_journals = None # 30347
        self.journal_embedding_size = 50

        self.num_hidden_layers = 1
        self.hidden_layer_size = 3365
        self.hidden_layer_act = 'relu'
        self.inputs_dropout_rate = 0.
        self.dropout_rate = 0.5

        self.output_layer_act = 'sigmoid'
        self.output_layer_size = 1
        
        self.init_threshold = 0.5
        self.init_learning_rate = 0.001

    @property
    def hidden_layer_sizes(self):
        return [self.hidden_layer_size]*self.num_hidden_layers

    @property
    def conv_filter_sizes(self):
        sizes = [self.min_conv_filter_size + self.conv_filter_size_step*idx for idx in range(self.num_conv_filter_sizes)]
        return sizes

    @property
    def conv_num_filters(self):
        num_filters = round(self.total_conv_filters / len(self.conv_filter_sizes))
        return num_filters

    @property
    def _pp_config(self):
        return self._parent.inputs.preprocessing

    @property
    def vocab_size(self):
        return self._pp_config.vocab_size

    @property
    def title_max_words(self):
        return self._pp_config.title_max_words

    @property
    def abstract_max_words(self):
        return self._pp_config.abstract_max_words

    @property
    def num_year_completed_time_periods(self):
        return self._pp_config.num_year_completed_time_periods

    @property
    def num_pub_year_time_periods(self):
        return self._pp_config.num_pub_year_time_periods


class _PreprocessingConfig(_ConfigBase):
    def _initialize(self, machine_config):

        self.unknown_word_index = 1
        self.unknown_journal_index = 0
        self.padding_index = 0
        self.title_max_words = 64
        self.abstract_max_words = 448
        self.vocab_size = 400000
        self.min_year_completed = None # 1965
        self.max_year_completed = None # 2018
        self.model_max_year = None # 2018
        self.min_pub_year = None # 1809 
        self.max_pub_year = None # 2018
        self.date_format = '%Y-%m-%d'

    @property
    def num_year_completed_time_periods(self):
        if self.max_year_completed and self.min_year_completed:
            num_year_completed_time_periods = 1 + self.max_year_completed - self.min_year_completed
        else:
            num_year_completed_time_periods = None
        return num_year_completed_time_periods

    @property
    def num_pub_year_time_periods(self):
        if self.max_pub_year and self.min_pub_year:
            num_pub_year_time_periods = 1 + self.max_pub_year - self.min_pub_year
        else:
            num_pub_year_time_periods = None
        return num_pub_year_time_periods


class _ProcessingConfig(_ConfigBase):
    def _initialize(self, machine_config):
                                                  
        self.use_multiprocessing = machine_config.use_multiprocessing                                
        self.workers = machine_config.workers                                                
        self.max_queue_size = machine_config.max_queue_size


class _ReduceLearningRateConfig(_ConfigBase):
    def _initialize(self, _):

        self.factor = 0.33
        self.patience = 1
        self.min_delta = 0.001


class _RestoreConfig(_ConfigBase):
     def _initialize(self, machine_config):
        super()._initialize(machine_config)
         
        self.sub_dir = ''
        self.model_json_filename = 'model.json'
        self.encoding = ENCODING
        self.model_checkpoint_dir = 'checkpoints'
        self.model_checkpoint_filename = 'best_model.hdf5'
        self.weights_only_checkpoint = True
        self.threshold = 0.5
        self.learning_rate = 0.001


class _PredConfig(_RestoreConfig, _ProcessingConfig):
    def _initialize(self, machine_config):
        super()._initialize(machine_config)

        self.batch_size = 128
        self.limit = 1000000000


class _ResumeConfig(_RestoreConfig):
     def _initialize(self, machine_config):
        super()._initialize(machine_config)

        self.enabled = False
        self.resume_checkpoint_filename = 'best_model_resume.hdf5'
        self.resume_logger_filename  = 'logs_resume.csv'    
        

class _SaveConfig(_ConfigBase):
    def _initialize(self, _):

        self.settings_filename = 'settings.json'
        self.model_json_filename = 'model.json'
        self.encoding = ENCODING
        self.model_img_filename = 'model.png'


class _TensorboardConfig(_ConfigBase):
    def _initialize(self, _):

        self.enabled = False
        self.dir = 'logs'
        self.write_graph = True


class _InputsConfig(_ConfigBase):
    def _initialize(self, machine_config):

        self.preprocessing = _PreprocessingConfig(self, machine_config)


class _OptimizeFscoreThresholdConfig(_ProcessingConfig):
    def _initialize(self, machine_config):
        super()._initialize(machine_config)
        
        self.enabled = True
        self.batch_size = 128
        self.limit = 1000000000
        self.metric_name = 'fscore'
        self.alpha = 0.005
        self.k = 3


class _TrainingConfig(_ProcessingConfig):
    def _initialize(self, machine_config):
        super()._initialize(machine_config)

        self.batch_size = 128
        self.initial_epoch = 0
        self.max_epochs = 500
        self.train_limit = 1000000000
        self.dev_limit = 1000000000
        self.monitor_metric = 'val_fscore'
        self.monitor_mode = 'max'
        self.save_config = _SaveConfig(self, machine_config)
        self.optimize_fscore_threshold = _OptimizeFscoreThresholdConfig(self, machine_config)
        self.reduce_learning_rate = _ReduceLearningRateConfig(self, machine_config)
        self.early_stopping = _EarlyStoppingConfig(self, machine_config)
        self.tensorboard = _TensorboardConfig(self, machine_config)
        self.csv_logger = _CsvLoggerConfig(self, machine_config)
        self.resume = _ResumeConfig(self, machine_config)


class Config(_ConfigBase):
    def __init__(self, machine_config):
        super().__init__(self, machine_config)

    def _initialize(self, machine_config):

        self.root_dir = machine_config.runs_dir
        self.data_dir = machine_config.data_dir
        self.inputs = _InputsConfig(self, machine_config)
        self.model = _ModelConfig(self, machine_config)
        self.train = _TrainingConfig(self, machine_config)
        self.pred_config = _PredConfig(self, machine_config)