import json


class Parameters(object):
    def __init__(self, parameter_name, parameter_func):
        self.parameter_name = parameter_name
        self.parameter_func = parameter_func
        self.parameter_value = None

    def p_name(self):
        return self.parameter_name

    def p_value(self):
        return self.parameter_value

    def get_from_source(self, source, default):
        value = source.pop(self.p_name(), default)
        if value is None:
            return None
        self.parameter_value = self.parameter_func(value)
        return self.parameter_value


class FloatParameter(Parameters):
    def __init__(self, parameter_name):
        super(FloatParameter, self).__init__(parameter_name, lambda value: float(value))


class BoolParameter(Parameters):
    def __init__(self, parameter_name):
        super(BoolParameter, self).__init__(parameter_name, lambda value: bool(value))


class IntParameter(Parameters):
    def __init__(self, parameter_name):
        super(IntParameter, self).__init__(parameter_name, lambda value: int(round(float(value))))


class StringParameter(Parameters):
    def __init__(self, parameter_name):
        super(StringParameter, self).__init__(parameter_name, lambda value: str(value))


class DictParameter(Parameters):
    def __init__(self, parameter_name):
        super(DictParameter, self).__init__(parameter_name,
                                            lambda value: json.loads(value) if value is not None else None)


class MonotonicityParameter(DictParameter):
    def __init__(self):
        super(MonotonicityParameter, self).__init__('monotonicity')

    @staticmethod
    def get_monotone_constraints(value, columns):
        if value is None:
            return None
        mc = [str(value.get(col, 0)) for c, col in enumerate(columns)]
        if len(mc) == 0:
            return None
        return ','.join(mc)

    @staticmethod
    def get_monotone_constraints_method(value):
        if value is None:
            return 'basic'
        return value.get('monotone_constraints_method', 'basic')

    @staticmethod
    def get_monotone_penalty(value):
        if value is None:
            return 0.0
        return value.get('monotone_penalty', 0.0)


class CategoricalParameter(StringParameter):
    def __init__(self, parameter_name):
        super(CategoricalParameter, self).__init__(parameter_name)

    def get_from_source(self, source, default):
        return self.p_name()

    @staticmethod
    def extract_from_source(source, marker='categorical'):
        keys = [k for k, v in source.items() if isinstance(v, str) and v == marker]
        parameters = list()
        for key in keys:
            source.pop(key)
            parameters.append(CategoricalParameter(key).p_name())
        return parameters


class ParametersBuffer:
    def __init__(self):
        self._boosting_type = StringParameter('boosting_type')
        self._num_leaves = IntParameter('num_leaves')
        self._max_depth = IntParameter('max_depth')
        self._learning_rate = FloatParameter('learning_rate')
        self._n_estimators = IntParameter('n_estimators')
        self._subsample_for_bin = IntParameter('subsample_for_bin')
        self._objective = StringParameter('objective')
        self._eval_metric = StringParameter('eval_metric')
        self._metrics = StringParameter('metrics')
        self._min_split_gain = FloatParameter('min_split_gain')
        self._min_child_weight = FloatParameter('min_child_weight')
        self._min_child_samples = IntParameter('min_child_samples')
        self._subsample = FloatParameter('subsample')
        self._subsample_freq = IntParameter('subsample_freq')
        self._random_state = IntParameter('random_state')
        self._colsample_bytree = FloatParameter('colsample_bytree')
        self._reg_alpha = FloatParameter('reg_alpha')
        self._reg_lambda = FloatParameter('reg_lambda')
        self._class_weight = DictParameter('class_weight')
        self._n_jobs = IntParameter('n_jobs')
        self._early_stopping_rounds = IntParameter('early_stopping_rounds')
        self._importance_type = StringParameter('importance_type')
        self._target = StringParameter('target')
        self._silent = BoolParameter('silent')
        self._is_unbalance = BoolParameter('is_unbalance')
        self._scale_pos_weight = FloatParameter('scale_pos_weight')
        self._max_importance_features = IntParameter('max_importance_features')
        self._true_accuracy = FloatParameter('true_accuracy')
        self._monotonicity = MonotonicityParameter()

    @property
    def boosting_type(self):
        return self._boosting_type

    @property
    def num_leaves(self):
        return self._num_leaves

    @property
    def max_depth(self):
        return self._max_depth

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def n_estimators(self):
        return self._n_estimators

    @property
    def subsample_for_bin(self):
        return self._subsample_for_bin

    @property
    def objective(self):
        return self._objective

    @property
    def eval_metric(self):
        return self._eval_metric

    @property
    def metrics(self):
        return self._metrics

    @property
    def min_split_gain(self):
        return self._min_split_gain

    @property
    def min_child_weight(self):
        return self._min_child_weight

    @property
    def min_child_samples(self):
        return self._min_child_samples

    @property
    def subsample(self):
        return self._subsample

    @property
    def subsample_freq(self):
        return self._subsample_freq

    @property
    def random_state(self):
        return self._random_state

    @property
    def colsample_bytree(self):
        return self._colsample_bytree

    @property
    def reg_alpha(self):
        return self._reg_alpha

    @property
    def reg_lambda(self):
        return self._reg_lambda

    @property
    def class_weight(self):
        return self._class_weight

    @property
    def n_jobs(self):
        return self._n_jobs

    @property
    def early_stopping_rounds(self):
        return self._early_stopping_rounds

    @property
    def importance_type(self):
        return self._importance_type

    @property
    def target(self):
        return self._target

    @property
    def silent(self):
        return self._silent

    @property
    def is_unbalance(self):
        return self._is_unbalance

    @property
    def scale_pos_weight(self):
        return self._scale_pos_weight

    @property
    def max_importance_features(self):
        return self._max_importance_features

    @property
    def true_accuracy(self):
        return self._true_accuracy

    @property
    def monotonicity(self):
        return self._monotonicity
