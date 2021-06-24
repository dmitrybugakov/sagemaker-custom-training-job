import json
import logging
import os
import sys
import traceback

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from parameters import ParametersBuffer, CategoricalParameter, MonotonicityParameter
from sklearn.metrics import r2_score, mean_absolute_error as mae

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

VALIDATION_DATA_TAG = "validation"


class Boosting(object):
    def __init__(self,
                 is_hyp_parameters_tuning=False,
                 base_path=None,
                 train_channel='train',
                 validation_channel='validation',
                 chunk_size=100,
                 model_file_name='model',
                 pkl_data=False,
                 compression='infer',
                 allow_post_validation=True):
        if base_path is None:
            self.base_path = '/opt/ml/'
        else:
            self.base_path = base_path
        self.input_path = os.path.join(self.base_path, 'input/data')
        self.model_path = os.path.join(self.base_path, 'model')
        self.output_path = os.path.join(self.base_path, 'output')
        # noinspection SpellCheckingInspection
        self.config_path = os.path.join(self.base_path, 'input', 'config')
        self.param_path = os.path.join(self.base_path, self.config_path, 'hyperparameters.json')
        self.chunk_size = chunk_size
        self.train_channel = train_channel
        self.validation_channel = validation_channel
        self.model_file_name = model_file_name
        self.validation_file_name = 'validation.json'
        self.p_buffer = ParametersBuffer()
        self.is_hyp_parameters_tuning = is_hyp_parameters_tuning
        self.pkl_data = pkl_data
        self.compression = compression
        self.allow_post_validation = allow_post_validation

    def load_parameters(self, param_path):
        with open(param_path, 'r') as tc:
            tp = json.load(tc)
        log.info('Parameters {}'.format(tp))
        target = self.p_buffer.target.get_from_source(tp, 'var1(t)')
        excluded_keys = CategoricalParameter.extract_from_source(tp, marker='excluded')
        categorical_keys = CategoricalParameter.extract_from_source(tp, marker='categorical')
        log.info('Features (cat.): {}'.format(categorical_keys))
        metrics = self.p_buffer.metrics.get_from_source(tp, 'l1')
        metrics = metrics.strip().split(',')
        metrics = [m.strip() for m in metrics]
        log.info('Training metrics: {}'.format(metrics))
        objective = self.p_buffer.objective.get_from_source(tp, 'regression')
        eval_metric = self.p_buffer.eval_metric.get_from_source(tp, 'l1')
        early_stopping_rounds = self.p_buffer.early_stopping_rounds.get_from_source(tp, 10)
        max_importance_features = self.p_buffer.max_importance_features.get_from_source(tp, 10)
        true_accuracy = self.p_buffer.true_accuracy.get_from_source(tp, 1e-3)
        params = {
            self.p_buffer.boosting_type.p_name(): self.p_buffer.boosting_type.get_from_source(tp, 'gbdt'),
            self.p_buffer.num_leaves.p_name(): self.p_buffer.num_leaves.get_from_source(tp, 31),
            self.p_buffer.max_depth.p_name(): self.p_buffer.max_depth.get_from_source(tp, -1),
            self.p_buffer.learning_rate.p_name(): self.p_buffer.learning_rate.get_from_source(tp, 0.1),
            self.p_buffer.n_estimators.p_name(): self.p_buffer.n_estimators.get_from_source(tp, 100),
            self.p_buffer.subsample_for_bin.p_name(): self.p_buffer.subsample_for_bin.get_from_source(tp, 200000),
            self.p_buffer.objective.p_name(): objective,
            self.p_buffer.metrics.p_name(): metrics,
            self.p_buffer.class_weight.p_name(): None,
            self.p_buffer.min_split_gain.p_name(): self.p_buffer.min_split_gain.get_from_source(tp, 0.0),
            self.p_buffer.min_child_weight.p_name(): self.p_buffer.min_child_weight.get_from_source(tp, 1e-3),
            self.p_buffer.min_child_samples.p_name(): self.p_buffer.min_child_samples.get_from_source(tp, 20),
            self.p_buffer.subsample.p_name(): self.p_buffer.subsample.get_from_source(tp, 1.0),
            self.p_buffer.subsample_freq.p_name(): self.p_buffer.subsample_freq.get_from_source(tp, 0),
            self.p_buffer.colsample_bytree.p_name(): self.p_buffer.colsample_bytree.get_from_source(tp, 1.0),
            self.p_buffer.reg_alpha.p_name(): self.p_buffer.reg_alpha.get_from_source(tp, 0.0),
            self.p_buffer.reg_lambda.p_name(): self.p_buffer.reg_lambda.get_from_source(tp, 0.0),
            self.p_buffer.random_state.p_name(): self.p_buffer.random_state.get_from_source(tp, None),
            self.p_buffer.n_jobs.p_name(): self.p_buffer.n_jobs.get_from_source(tp, -1),
            self.p_buffer.importance_type.p_name(): self.p_buffer.importance_type.get_from_source(tp, 'split'),
            self.p_buffer.silent.p_name(): self.p_buffer.silent.get_from_source(tp, True),
            self.p_buffer.is_unbalance.p_name(): self.p_buffer.is_unbalance.get_from_source(tp, None),
            self.p_buffer.scale_pos_weight.p_name(): self.p_buffer.scale_pos_weight.get_from_source(tp, None),
            self.p_buffer.monotonicity.p_name(): self.p_buffer.monotonicity.get_from_source(tp, None),
        }
        return (target, excluded_keys, categorical_keys, eval_metric,
                early_stopping_rounds, max_importance_features, true_accuracy,
                self.p_buffer.importance_type.p_value(),
                params)

    @staticmethod
    def load_data(input_path, chunk_size, channel, target, excluded_keys=None, pkl_data=False, compression='infer'):
        data_path = os.path.join(input_path, channel)
        ext = '.csv' if not pkl_data else '.pkl'
        input_files = [os.path.join(data_path, file_path) for file_path in os.listdir(data_path)
                       if file_path.endswith(ext)]
        if len(input_files) == 0:
            raise ValueError('There are no files in {}.'.format(data_path))

        if pkl_data:
            raw_data_it = [pd.read_pickle(file_path) for file_path in input_files]
        else:
            raw_data_it = [pd.read_csv(file_path, chunksize=chunk_size, compression=compression) for file_path in
                           input_files]
        raw_data = []
        for it in raw_data_it:
            if pkl_data:
                raw_data.append(it)
            else:
                for chunk in it:
                    raw_data.append(chunk)
        data = pd.concat(raw_data)
        y = data[target].values
        if excluded_keys is not None and len(excluded_keys) > 0:
            excluded_keys.append(target)
        else:
            excluded_keys = [target]
        x = data.drop(excluded_keys, axis=1)
        return x, y, excluded_keys

    def plot(self, bst, plot_type,
             allow_zeros=False,
             max_features=20,
             precision=4):
        try:
            lgb.plot_importance(bst,
                                importance_type=plot_type,
                                max_num_features=max_features,
                                ignore_zero=~allow_zeros,
                                figsize=(13, 9),
                                precision=precision)
            plt.savefig('{}/features.png'.format(self.model_path))
        except ValueError as ve:
            if ve is None or 'not enough values to unpack' not in str(ve):
                raise ve

    def validate(self, gbm, train_x, train_y, val_x, val_y):
        train_y_predicted = gbm.booster_.predict(train_x)
        log.info('Train predicted: {}'.format(str(train_y_predicted)))
        val_y_predicted = gbm.booster_.predict(val_x)
        log.info('Val predicted: {}'.format(str(val_y_predicted)))

        validation = {
            'train_r2': r2_score(train_y, train_y_predicted),
            'val_r2': r2_score(val_y, val_y_predicted),
            'train_mae': mae(train_y, train_y_predicted),
            'val_mae': mae(val_y, val_y_predicted)
        }
        with open(os.path.join(self.model_path, self.validation_file_name), 'w') as f:
            json.dump(validation, f)
        log.info('VALIDATION:{}'.format(json.dumps(validation)))

    def store_excluded_keys(self, excluded_keys):
        with open(os.path.join(self.model_path, 'excluded_keys.json'), 'w') as f:
            json.dump(excluded_keys, f)
        log.info('EXCLUDED_KEYS:{}'.format(json.dumps(excluded_keys)))

    def train(self):
        log.info('Gradient boosting training')
        try:
            (target, excluded_keys, categorical_keys, eval_metric, early_stopping_rounds,
             max_importance_features, true_accuracy, importance_type, params) = self.load_parameters(self.param_path)

            train_x, train_y, train_ek = self.load_data(self.input_path, self.chunk_size, self.train_channel, target,
                                                        excluded_keys=excluded_keys, pkl_data=self.pkl_data,
                                                        compression=self.compression)
            val_x, val_y, val_ek = self.load_data(self.input_path, self.chunk_size, self.validation_channel, target,
                                                  excluded_keys=excluded_keys, pkl_data=self.pkl_data,
                                                  compression=self.compression)

            monotonicity = params.pop('monotonicity')
            params['monotone_constraints'] = MonotonicityParameter.get_monotone_constraints(monotonicity,
                                                                                            train_x.columns)
            params['monotone_constraints_method'] = MonotonicityParameter.get_monotone_constraints_method(monotonicity)
            params['monotone_penalty'] = MonotonicityParameter.get_monotone_penalty(monotonicity)
            log.info(f'Parameters: {params}')
            gbm = lgb.LGBMRegressor(**params)
            gbm.fit(
                train_x,
                train_y,
                eval_set=[(val_x, val_y)],
                eval_metric=eval_metric,
                eval_names=[VALIDATION_DATA_TAG],
                early_stopping_rounds=early_stopping_rounds,
                categorical_feature=[c for c, col in enumerate(train_x.columns) if col in categorical_keys],
                verbose=True)

            gbm.booster_.save_model(os.path.join(self.model_path, self.model_file_name))
            self.plot(gbm,
                      plot_type=importance_type,
                      max_features=min(max_importance_features, len(train_x.columns)))
            self.store_excluded_keys(train_ek)
            self.validate(gbm, train_x, train_y, val_x, val_y)
            print("{} best scores: ".format(VALIDATION_DATA_TAG), gbm.best_score_)
            print("{} best score (l1): ".format(VALIDATION_DATA_TAG),
                  gbm.best_score_.get(VALIDATION_DATA_TAG).get('l1'))

            return gbm
        except Exception as e:
            trc = traceback.format_exc()
            with open(os.path.join(self.output_path, 'failure'), 'w') as s:
                s.write('Training exception: ' + str(e) + '\n' + trc)
            log.error('Training exception: ' + str(e) + '\n' + trc, file=sys.stderr)
            sys.exit(255)
