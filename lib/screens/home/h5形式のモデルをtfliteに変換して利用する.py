'''ライブラリのインストール'''
#pip install -q pyyaml h5py # HDF5フォーマットでモデルを保存するために必要


'''ライブラリのインポート'''
import tensorflow as tf
from tensorflow import keras
 
if __name__ == "__main__":
    print(tf.version.VERSION)



'''サンプルデータセットの取得'''
#実際に学習させたモデルを保存・復元します。
#そのため、そのモデルを学習させるためにデータを用意する必要があります。

#ここは良く使われる「MNIST dataset 」の出番です。
#ただし、全部は使わずに最初の1000件とします。
#あくまで、モデルの保存・復元の検証がメインです。

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
 
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
 
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


'''モデルの定義・学習'''
#まずは、モデル定義の関数を作成します。

# 短いシーケンシャルモデルを返す関数
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
  ])
   
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
   
  return model
#上記で作成した関数を利用して、モデルを定義。
#そのモデルにMNISTのサンプルデータを食わせます。


'''新しいモデルのインスタンスを作成して訓練'''
#新しいモデルのインスタンスを作成
model = create_model()
# モデルの構造を表示
model.summary()
#訓練
model.fit(train_images, train_labels, epochs=5)



'''モデルの保存'''
#事前にsaved_modelのディレクトリを作成しておいてください。
# モデル全体を SavedModel として保存
#!mkdir -p saved_model

'''モデルの保存形式には、次の2種類があります。'''
#SavedModel フォーマット
#HDF5ファイル

'''SavedModel フォーマット'''
#model.save('saved_model/my_model')
#上記で作成すると、saved_modelディレクトリの下にmy_modelディレクトリが作成されます。
model.save('saved_model/my_model')




'''SavedModel を読み込んで新しい Keras モデルを作成します。'''
new_model = tf.keras.models.load_model('saved_model/my_model')

# アーキテクチャを確認
new_model.summary()


'''リストアされたモデルは元のモデルと同じ引数を用いてコンパイルされます。読み込んだモデルを用いて評価と予測を行ってみましょう。'''
# リストアされたモデルを評価
loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

print(new_model.predict(test_images).shape)









'''HDF5ファイル'''
# 新しいモデルのインスタンスを作成して訓練
#model = create_model()
#model.fit(train_images, train_labels, epochs=5)

# HDF5 ファイルにモデル全体を保存
# 拡張子 '.h5' はモデルが HDF5 で保存されているということを暗示する
#model.save('assets/stockcard.h5')
model.save('saved_model/my_model.h5')


'''保存したファイルを使ってモデルを再作成します。'''
# 同じモデルを読み込んで、重みやオプティマイザーを含むモデル全体を再作成
new_model = tf.keras.models.load_model('saved_model/my_model.h5')

# モデルのアーキテクチャを表示
new_model.summary()

#正解率を検査します。
loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))




'''h5形式のモデルをtfliteに変換して利用する
keras model to tflite with python
pythonで行う。tfliteへの変換スクリプトは以下'''

import tensorflow as tf

# load model
#model = tf.keras.models.load_model('./models/model.hdf5')
model = tf.keras.models.load_model('saved_model/my_model.h5')
# convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# save
open("saved_model/model.tflite", "wb").write(tflite_model)




'''


'''


# HDF5 ファイルにモデル全体を保存
# 拡張子 '.h5' はモデルが HDF5 で保存されているということを暗示する
model.save('saved_model/my_model.h5')
#上記で作成すると、saved_modelディレクトリの下にmy_model.h5ファイルが作成されます。
#saved_modelディレクトリには、以下の状態です。
#SavedModel フォーマットのところで作成した、ディレクトリがありますね。


'''#ここまでのコードをまとめると、以下となります。'''

import tensorflow as tf
from tensorflow import keras
 
# 短いシーケンシャルモデルを返す関数
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
  ])
   
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
   
  return model
 
if __name__ == "__main__":
     
    print(tf.version.VERSION)
     
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
     
    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]
     
    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
     
    # 新しいモデルのインスタンスを作成して訓練
    model = create_model()
    model.fit(train_images, train_labels, epochs=5)
     
    # モデル全体を SavedModel として保存
    model.save('saved_model/my_model')
     
    # HDF5 ファイルにモデル全体を保存
    # 拡張子 '.h5' はモデルが HDF5 で保存されているということを暗示する
    #model.save('saved_model/my_model.h5')

 
'''モデルの復元'''
#復元自体は、次のコードで簡単にできます。

#SavedModel フォーマットなら作成したディレクトリ「saved_model/my_model」を指定します。
new_model = tf.keras.models.load_model('saved_model/my_model')

#HDF5ファイルなら作成したファイル「saved_model/my_model.h5」を指定します。
new_model = tf.keras.models.load_model('saved_model/my_model.h5')

 
'''復元したモデルの利用'''
#復元したものを利用してこそ、モデルを保存する意味があります。
#保存したモデルを利用しているコードは以下。

import tensorflow as tf
 
if __name__ == "__main__":
     
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
     
    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]
     
    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
     
    # 同じモデルを読み込んで、重みやオプティマイザーを含むモデル全体を再作成
    new_model = tf.keras.models.load_model('saved_model/my_model')
    #new_model = tf.keras.models.load_model('saved_model/my_model.h5')
     
    # 正解率を検査
    loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))

'''このコードの実行結果は以下。'''
#32/32 - 0s - loss: 0.4259 - accuracy: 0.0840
#Restored model, accuracy:  8.40%
#かなり低い正解率ですが、利用できていますね。
#結果はどうあれ、モデルの保存・利用はこれでOK。










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