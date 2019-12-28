from flask import Flask, request, make_response, jsonify, render_template, request, session, redirect, flash
import json
import numpy as np
import os
import pandas as pd
import werkzeug
import MeCab

from sqlalchemy import create_engine, Column, Integer, String, Text, MetaData, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from model import Config
from db_setting import ENGINE, Base

from joblib import dump, load
from datetime import datetime
from pathlib import Path
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report

app = Flask(__name__)

app.secret_key = 'secret key'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024

MODEL_DIR = 'model'


@app.route('/', methods=['GET'])
def top():
    ses = Session()
    configs = ses.query(Config.name).all()
    configs = sorted([str(config[0]) for config in configs])
    return render_template('index.html', names=configs)

@app.route('/upload', methods=['GET'])
def upload_viewer():
    return render_template('upload.html')


@app.route('/data_list', methods=['GET'])
def data_list():
    return render_template('list.html')


@app.route('/model_config', methods=['GET'])
def model():
    return render_template('model.html')


@app.route('/model_learn', methods=['GET'])
def model_learn():
    setup = {'name': 'モデル学習', 'do_action': '/learn_model'}
    forms = get_config_data()
    return render_template('model_learn.html', forms=forms, setup=setup)


@app.route('/learn_model', methods=['POST'])
def learn_model():
    form = request.form
    name = form['name']
    if not(name):
        name = form['config'] + '_' + form['data']
    model, X, label, vectorizer = ml_setup(form)
    model.fit(X, label)
    if not(os.path.exists(os.path.join(MODEL_DIR, name))):
        os.makedirs(os.path.join(MODEL_DIR, name))
    model_save_file =os.path.join(MODEL_DIR, name, 'model.joblib')
    feature_save_file = os.path.join(MODEL_DIR, name, 'feature.joblib')
    with open(feature_save_file, 'wb') as f:
        dump(vectorizer, f)

    with open(model_save_file, 'wb') as f:
        dump(model, f)
        
    flash('モデルの学習に成功しました', "alert alert-success")
    return redirect('/model_learn')


@app.route('/model_setup', methods=['GET'])
def setup_model():
    setup = {'name': 'モデル読み込み', 'do_action': '/do_load_model'}
    model_dir = Path(MODEL_DIR)
    models = model_dir.glob("*")
    models = sorted([str(model.stem) for model in models])
    model_param = {'name': 'model', 'label': 'sel1', 'files': models}
    forms = [model_param]
    return render_template('model_load.html', forms=forms, setup=setup)


@app.route('/do_load_model', methods=['POST'])
def load_model():
    global model
    global vectorizer
    model_name = request.form['model']
    model_path = os.path.join(MODEL_DIR, model_name, 'model.joblib')
    vectorizer_path = os.path.join(MODEL_DIR, model_name, 'feature.joblib')
    with open(vectorizer_path, 'rb') as f:
        vectorizer = load(f)

    with open(model_path, 'rb') as f:
        model = load(f)

    flash('モデルの読み込みに成功しました', "alert alert-success")
    return redirect('/model_setup')


@app.route('/valification', methods=['GET'])
def valify_setup():
    setup= {'name': '交差検証', 'do_action': '/do_validate'}
    forms = get_config_data()
    return render_template('validate.html', forms=forms, setup=setup)


@app.route('/do_validate', methods=['POST'])
def do_validate():
    form = request.form
    k = int(form['split_n'])
    kfold = KFold(n_splits=k)
    model, X, label = ml_setup(form)
    score = cross_val_score(model, X, label, cv=kfold)
    result = np.mean(score)
    return render_template('result.html', score=result)


@app.route('/predict', methods=['POST'])
def predict():
    response = {
        "success": False,
        "Content-Type": "application/json"
    }
    if request.method == "POST":
        if request.get_json().get("text"):
            print(type(model))
            print(type(vectorizer))
            text = request.get_json().get("text")
            wakati = get_wakati(text)
            feature = vectorizer.transform([wakati])
            result = model.predict(feature)
            response["prediction"] = result.tolist()
            response["success"] = True
    return jsonify(response)

            
@app.route('/upload_data', methods=['POST'])
def upload_multipart():
    if 'uploadfile' not in request.files:
        flash("ファイルが指定されていません", "alert alert-danger")
        return redirect('/upload')
        make_response(jsonify({'result': 'uploadfile is required.'}))

    file = request.files['uploadfile']
    filename = request.form['filename']
    if not(filename):
        flash('ファイル名が指定されていません', "alert alert-danger")
        return redirect('/upload')

    df = pd.read_csv(file, header=0)
    print(df.head(3))
    df.to_sql(filename, ENGINE, index=False)
    flash("ファイルをデータベースに保存しました", "alert alert-success")
    return redirect('/upload')


@app.route('/file_list', methods=['GET'])
def get_file_list():
    meta = MetaData(bind=ENGINE, reflect=True)
    tables = list(meta.tables)
    datas = sorted([str(table) for table in tables if table != 'config'])
    return render_template('files.html', names=datas)


@app.route('/save_config', methods=['POST'])
def save_model_config():
    config_name = request.form['name']
    if not(config_name):
        flash('config名を指定してください', "alert alert-danger")
        return redirect('/model_config')
    model_name = request.form['algo']
    hyper_parameter = request.form['hyper']
    ses = Session()
    ses.add(Config(name=config_name, algo=model_name, hyper=hyper_parameter))
    flash('登録に成功しました', "alert alert-success")
    ses.commit()
    ses.close()
    return redirect('/model_config')


@app.errorhandler(werkzeug.exceptions.RequestEntityTooLarge)
def handle_over_max_file_size(error):
    return 'result : file size is overed'


def ml_setup(form):
    ses = Session()
    config_name = form['config']
    config_obj = ses.query(Config).filter(Config.name == config_name).one()
    config = config_obj.toDict()
    
    algo = config['algo']
    if len(config['hyper']) != 0:
        hyper_params = json.loads(config['hyper'])
    else:
        hyper_params = None

    if algo == "KNN":
        model_type = KNeighborsClassifier
    elif algo == "Linear SVM":
        model_type = LinearSVC
    elif algo == "Kernel SVM":
        model_type = SVC
    elif algo == "MLP":
        model_type = MLPClassifier
    elif algo == "Random Forest":
        model_type = RandomForestClassifier
    else:
        raise ValueError("{} is not defined".format(model_name))

    if hyper_params:
        model = model_type(**hyper_params)
    else:
        model = model_type()

    data_name = form['data']
    data = pd.read_sql_table(data_name, ENGINE)
    print(data.columns)
    text_field = form['text_field']
    text = data.loc[:, text_field].values
    label = data.loc[:, 'annotation'].values

    vectorizer = CountVectorizer()
    docs = []
    for t in text:
        wakati = get_wakati(t)
        docs.append(wakati)
    X = vectorizer.fit_transform(docs)
    return (model, X, label, vectorizer)


def get_wakati(text):
    mecab = MeCab.Tagger('-O wakati')
    mecab.parse('')
    return mecab.parse(text)


def get_config_data():
    ses = Session()
    configs = ses.query(Config.name).all()
    configs = sorted([str(config[0]) for config in configs])
    config_param = {'name': 'config', 'label': 'sel1', 'files': configs}
    meta = MetaData(bind=ENGINE, reflect=True)
    tables = list(meta.tables)
    datas = sorted([str(table) for table in tables if table != 'config'])
    data_param = {'name': 'data', 'label': 'sel2', 'files': datas}
    forms = [config_param, data_param]
    ses.close()
    return forms


if __name__ == "__main__":
    print(app.url_map)
    Session = sessionmaker(bind=ENGINE)
    app.run(host='localhost', port=8000, debug=True)
