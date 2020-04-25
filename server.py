# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras import backend as K
from PIL import Image
import os, io, base64

app = Flask(__name__)

classes =[
'aishimeji','aitake','akachishiotake','akamomitake','akatake','akayamadori','akayamatake','akebonoawatake','akebonosakurashimeji','amigasatake','amihanaiguchi','amitake','anzutake','ashibeniiguchi','awatake','benihidatake','beninaginatatake','benitengutake','bunaharitake','bunashimeji','chichiawatake','chichitake','chishiotake','dokubenitake','dokutsurutake','enokitake','erimakitsuchiguri','eringi','eseorimiki','fukurotsurutake','fuyuyamatake','gantake','gomutake','hanabiranikawatake','hanabiratake','hanagasatake','hanahoukitake','hanaiguchi','hanaochibatake','harushimeji','hatakeshimeji','hatsutake','hebikinokomodoki','hiirotake','hiirotyawantake','hinanohigasa','hiratake','hitokuchitake','hitoyotake','hokoritake','honshimeji','houkitake','inusenbontake','irogawari','itachitake','kaentake','kakishimeji','kanbatake','kanoshita','kanzoutake','karahatsutake','karakasatake','kawaratake','kawarihatsu','kerouji','kiirosuppontake','kikurage','kinugasatake','kiraratake','kishimeji','kisoumentake','kitsunenoefude','kitsunenorousoku','kitsunetake','kofukisarunokoshikake','koganetake','koniroipponshimeji','kotengutakemodoki','koujitake','koumoritake','koutake','kuchibenitake','kugitake','kurifuusentake','kuritake','kurohanabiratake','kurohatsu','kurorappatake','kusaurabenitake','maitake','mamezayatake','mannentake','masutake','matsuouji','matsutake','mimibusatake','moegitake','morinofujiirotake','morinokarebatake','mukitake','murasakinaginatatake','murasakishimeji','murasakiyamadoritake','mursakifuusentake','musasabitake','nagaenosugitake','naginatatake','nameko','naratake','naratakemodoki','nigakuritake','ningyotake','nioikobenitake','nioushimeji','noboriryutake','noutake','numeriiguchi','numerisasatake','numerisugitake','numerisugitakemodoki','numeritsubatake','ohichoutake','ohshirokarakasatake','ohwaraitake','onifusube','oninaratake','onitake','oogomutake','oohouraitake','oowaraitake','oshiroishimeji','otomenokasa','otsunentakemodoki','ougitake','saketsubatake','sakurashimeji','sakuratake','samatsumodoki','sangoharitake','sankotake','sasakurehitoyotake','sasakureshiroonitake','sasatake','shiitake','shimofurishimeji','shirohatsu','shirohatsumodoki','shirokikurage','shironametsumutake','shiroonitake','shirosoumentake','shirotamagotengutake','sorairotake','suehirotake','sugiedatake','sugihiratake','sugitake','sumizomeyamaiguchi','suppontake','susukeyamadoritake','syagumaamigasatake','syakashimeji','syougenji','tamagotake','tamagotengutake','tamagotengutakemodoki','tamashiroonitake','tamaurabenitake','tamogitake','tengunomeshigai','tengutake','togariamigasatake','tonbimaitake','truffle','tsubaaburashimeji','tsuchiguri','tsuchikaburi','tsuchisugitake','tsukiyotake','tsurutake','tyakaigaratake','tyanametsumutake','ukonhatsu','urabenigasa','urabenihoteishimeji','usuhiratake','usutake','wakakusatake','yamabushitake','yamadoritake','yamadoritakemodoki','yamaiguchi','zaraenoharatake','zaraenohitoyotake'
]
num_classes = len(classes)
INPUTSIZE = 224

def img_pred(image):
    # 保存したモデルをロード
    model = load_model('vgg16_transfer.h5')

    # 読み込んだ画像を行列に変換
    img_array = img_to_array(image)
    # data = np.asarray(image)

    # 3次元を4次元に変換、入力画像は1枚なのでsamples=1
    img_dims = np.expand_dims(img_array, axis=0)

    # Top2のクラスの予測
    img_dims = img_dims / 255.0
    preds = model.predict(img_dims)[0]

    top = 5
    pred_labels = preds.argsort()[-top:][::-1]
    result_class = [classes[x] for x in pred_labels]
    result_pred = [preds[x] for x in pred_labels]

    K.clear_session()

    # resultsを整形
    return result_class, result_pred

@app.route('/')
def index():
    return render_template('./flask_api_index.html')

@app.route('/result', methods=['POST'])
def result():
    # submitした画像が存在したら処理する
    if request.files['image']:
        # 画像の読み込み
        image_load = load_img(request.files['image'], target_size=(INPUTSIZE,INPUTSIZE))
        buf = io.BytesIO()
        image_load.save(buf, 'png')
        qr_b64str = base64.b64encode(buf.getvalue()).decode("utf-8")
        qr_b64data = "data:image/png;base64,{}".format(qr_b64str)

        # クラスの予測をする関数の実行
        class_names, prediction = img_pred(image_load)

        pred1 = int(round(prediction[0],2)*100)
        pred2 = int(round(prediction[1],2)*100)
        pred3 = int(round(prediction[2],2)*100)
        pred4 = int(round(prediction[3],2)*100)
        pred5 = int(round(prediction[4],2)*100)
        rank1 = class_names[0]
        rank2 = class_names[1]
        rank3 = class_names[2]
        rank4 = class_names[3]
        rank5 = class_names[4]

        # render_template('./result.html')
        return render_template('./result.html', img_url = qr_b64data, rank1=rank1, rank2=rank2, rank3=rank3, rank4=rank4, rank5=rank5, pred1=pred1, pred2=pred2, pred3=pred3, pred4=pred4, pred5=pred5)

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
