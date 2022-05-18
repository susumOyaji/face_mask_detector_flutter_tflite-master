'''h5形式のモデルをtfliteに変換して利用する
keras model to tflite with python
pythonで行う。tfliteへの変換スクリプトは以下'''

import tensorflow as tf

# load model
model = tf.keras.models.load_model('./models/model.hdf5')

# convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# save
open("./models/model.tflite", "wb").write(tflite_model)





'''flutterのセットアップ'''
#flutterで予測モデルを読み込む

#モデルとラベル情報をassetsに入れておき、pubspec.yamlに書いておく必要がある
#以下ではラベルをassets/label.txt、モデルをassets/model.tfliteとしている

'''
...
flutter:
  assets:
    - assets/label.txt
    - assets/model.tflite
...
label.txtは各indexに対応するラベル名を改行して記述しておく

'''

#0=dog 1=catのようなクラスに対応するときは、以下のような中身になる
#dog
#cat


'''予測(Flutter)
#予測実行用スクリプト。画像のパスを渡せば予測結果が取得できる

    await Tflite.loadModel(model: "assets/model.tflite",
    labels: "assets/label.txt");
    // TODO imageバイナリで渡す方法でやってみたい
    // https://pub.dev/packages/tflite
    var output =  await Tflite.runModelOnImage(
      path: imagePath,
       );

    results = output;


画像のパスimage_pickerを使って取得できる

  Future imageConnect(ImageSource source) async {
    final PickedFile img = await picker.getImage(source: source);
    if (img != null) {
      imagePath = img.path;
      image = File(img.path);
      loading = true;

      setState(() {});
      classifyImage();
    }
  }
  '''