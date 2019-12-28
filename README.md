# doc_classifier
文書分類するサービス(勉強用)

## 動かすとき
1. データベーステーブルの作成
`$ python model.py$`

2. アプリの起動
`$ python app.py`

## 必要ライブラリ
- pandas
- sklearn
- numpy
- flask
- flask_pagenate
- joblib
- mecab-python3
- sqlalchemy

## やらなかったこと
1. configを変更できるようにする
2. text以外をカテゴリ変数や数値として使用する
3. データの適宜追加
4. dataへのprimary keyの設定(sqliteではできないので、unique indexにした)
5. MVC的な分解(大半がapp.pyに入っている)
6. 諸々のtest

## この先必要そうな知見
1. DB系統
1.1. セキュリティー
1.2. 速度
1.3. ORMの深めな理解(発行クエリと速度など)
2. JS系

