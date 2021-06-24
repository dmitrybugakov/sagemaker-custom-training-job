from __future__ import print_function

import os
from io import StringIO
from flask import Flask, request, make_response, Response
from flask_restful import Resource, Api
import pandas as pd
import lightgbm as lgb
from flask_jsonpify import jsonify

"""
This is the file that implements a flask server to do inferences.
"""

PREFIX = '/opt/ml/'
MODEL_PATH = os.path.join(PREFIX, 'model')


class ModelService(object):
    def __init__(self, model_path, model_file_name):
        self.model = lgb.Booster(model_file=os.path.join(model_path, model_file_name))

    def predict(self, data):
        return self.model.predict(data)

    def health(self):
        return self.model is not None


class ModelPingController(Resource):
    def __init__(self, model_service):
        self.model_service = model_service

    def get(self):
        return make_response(jsonify({
            'OK': self.model_service.health()
        }), 200 if self.model_service.health() else 404)


class ModelPredictController(Resource):
    def __init__(self, model_service):
        self.model_service = model_service

    def post(self):
        if request.content_type == 'application/json':
            data = request.data.decode('utf-8')
            s = StringIO(data)
            data = pd.read_json(s, orient='record')
        elif request.content_type == 'text/csv':
            data = request.data.decode('utf-8')
            s = StringIO(data)
            data = pd.read_csv(s)
        else:
            return Response(response='Invalid content type {}'.format(request.content_type),
                            status=415,
                            mimetype='text/plain')
        try:
            predictions = self.model_service.predict(data)
            return make_response(jsonify(list(predictions)), 200)
        except Exception as e:
            return make_response(jsonify({
                'message': str(e)
            }), 500)


def create_application():
    app = Flask(__name__)
    model_service = ModelService(MODEL_PATH, 'model')
    api = Api(app)
    api.add_resource(ModelPingController, '/ping', resource_class_args=(model_service,))
    api.add_resource(ModelPredictController, '/invocations', resource_class_args=(model_service,))
    return app
