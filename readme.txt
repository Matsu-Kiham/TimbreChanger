音色変更ツール

製作者：Matsu-Kiham
バージョン：1.00

音声データ（wavファイル）を読み込み，離散フーリエ変換を行い，
指定したオクターブの中で最も振幅が大きい周波数の音を正弦波，
あるいは矩形波として鳴らした音声データ（wavファイル）を出力します．

導入方法：
TimbreChanger.exeをダウンロードして実行してください．
なお，TimbreChanger.pyはソフトのソースコードです．

アプリの操作法：
まず，アプリ左側にある”音を拾う範囲（オクターブ）”という文字の下に
あるチェックボックスから，wavファイルに出力したい範囲のオクターブ
を選んでクリックしてチェックを入れてください．
また，”開始音”という文字の下にあるドロップダウンリストから，
各オクターブの開始音を選んでください．
例えば，音を拾う範囲として3オクターブと4オクターブを選択し，
開始音としてFを選んだ場合，出力されるwavファイルには，
F3～E4から最も振幅が大きい周波数の1音，
F4～E5から最も振幅が大きい周波数の1音が鳴るように書き込まれます．
音を拾う範囲と開始音を選んだら，”正弦波”あるいは”矩形波”のボタンを
クリックしてください．クリックすると，ファイルの選択画面が表示
されるので，変換したいwavファイルを選んで”開く”ボタンをクリック
してください．変換が始まると変換が終わるまでアプリの操作ができなく
なります．変換が終了すると，TimbreChanger.exeがあるフォルダに
”out.wav”ファイルが作成されます．


作者本人への連絡
matsukiham★gmail.com（★を@に変えてください）

更新履歴
2020/10/04　バージョン1.00公開