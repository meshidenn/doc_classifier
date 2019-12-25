from flask import Flask, request, make_response, jsonify, render_template, request, session, redirect, flash
import json
import numpy as np
import os
import pandas as pd
import werkzeug
import MeCab
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

UPLOAD_DIR = 'upload'
CONFIG_DIR = 'config'
MODEL_DIR = 'model'


@app.route('/', methods=['GET'])
def top():
    config_dir = Path(CONFIG_DIR)
    configs = config_dir.glob("*")
    configs = sorted([str(conf.stem) for conf in configs])
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
    config_dir = Path(CONFIG_DIR)
    configs = config_dir.glob("*")
    configs = sorted([str(conf.stem) for conf in configs])
    config = {'name': 'config', 'label': 'sel1', 'files': configs}
    data_dir = Path(UPLOAD_DIR)
    datas = data_dir.glob("*")
    datas = sorted([str(data.stem) for data in datas])
    data = {'name': 'data', 'label': 'sel2', 'files': datas}
    forms = [config, data]
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
        os.mkdir(os.path.join(MODEL_DIR, name))
    model_save_file =os.path.join(MODEL_DIR, name, 'model.joblib')
    feature_save_file = os.path.join(MODEL_DIR, name, 'feature.joblib')
    with open(feature_save_file, 'wb') as f:
        dump(vectorizer, f)

    with open(model_save_file, 'wb') as f:
        dump(model, f)
        
    flash('モデルの学習に成功しました')
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

    flash('モデルの読み込みに成功しました')
    return redirect('/model_setup')


@app.route('/valification', methods=['GET'])
def valify_setup():
    setup= {'name': '交差検証', 'do_action': '/do_validate'}
    config_dir = Path(CONFIG_DIR)
    configs = config_dir.glob("*")
    configs = sorted([str(config.stem) for config in configs])
    config_param = {'name': 'config', 'label': 'sel1', 'files': configs}
    data_dir = Path(UPLOAD_DIR)
    datas = data_dir.glob("*")
    datas = sorted([str(data.stem) for data in datas])
    data_param = {'name': 'data', 'label': 'sel2', 'files': datas}
    forms = [config_param, data_param]
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

            
@app.route('/upload_result', methods=['POST'])
def upload_multipart():
    if 'uploadfile' not in request.files:
        flash("ファイルが指定されていません", "alert alert-danger")
        return render_template('upload.html')
        make_response(jsonify({'result': 'uploadfile is required.'}))

    file = request.files['uploadfile']
    filename = request.form['filename']
    if not(filename):
        flash('ファイル名が指定されていません', "alert alert-danger")
        return render_template('upload.html')

    savefilename = werkzeug.utils.secure_filename(filename)

    file.save(os.path.join(UPLOAD_DIR, savefilename))
    flash("設定を保存しました", "alert-success")
    return redirect('/upload')


@app.route('/file_list', methods=['GET'])
def get_file_list():
    target_dir = Path(UPLOAD_DIR)
    filelist = target_dir.glob('*')
    filelist = [str(fname.name) for fname in filelist]
    return render_template('files.html', names=filelist)


@app.route('/save_config', methods=['POST'])
def save_model_config():
    file_name = request.form['name']
    if not(file_name):
        flash('ファイル名を指定してください')
        return redirect('/model_config')
    file_name += '.json'
    model_name = request.form['algo']
    hyper_parameter = request.form['hyper']
    if hyper_parameter:
        parameter = json.loads(hyper_parameter)
    else:
        parameter = dict()
    parameter['algo'] = model_name
    filename = os.path.join(CONFIG_DIR, file_name)
    with open(filename, 'w') as f:
        json.dump(parameter, f)

    flash('登録に成功しました')
    return redirect('/model_config')


@app.errorhandler(werkzeug.exceptions.RequestEntityTooLarge)
def handle_over_max_file_size(error):
    return 'result : file size is overed'


def ml_setup(form):
    config_file = os.path.join(CONFIG_DIR, form['config'] + '.json')
    with open(config_file, 'r') as f:
        config = json.load(f)

    algo = config['algo']
    if 'hyper' in config:
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

    data_name = os.path.join(UPLOAD_DIR, form['data'])
    data = pd.read_csv(data_name)
    text_field = form['text_field']
    text = data.loc[:, text_field].values
    label = data.loc[:, 'label'].values

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


if __name__ == "__main__":
    print(app.url_map)
    app.run(host='localhost', port=8000, debug=True)
