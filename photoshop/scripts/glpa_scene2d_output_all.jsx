// ダイアログで出力先のフォルダを選択
var outputPath = Folder.selectDialog("出力先フォルダを選択");

var tempNum = app.activeDocument.layers.length;

// ユーザーがキャンセルボタンを押した場合は処理を中断
if (!outputPath) {
    alert("キャンセルされました。処理を中止します。");
} else {
    // すべてのレイヤーを非表示にする
    for (var i = 0; i < app.activeDocument.layers.length; i++) {
        app.activeDocument.layers[i].visible = false;
    }

    // レイヤーごとに処理
    for (var i = 0; i < app.activeDocument.layers.length; i++) {
        var layer = app.activeDocument.layers[i];
        layer.visible = true;  // レイヤーを表示

        // 画像が配置されている範囲を取得
        var boundsBefore = layer.bounds;
        var leftBefore = boundsBefore[0].value;
        var topBefore = boundsBefore[1].value;

        // 画像が配置されている範囲の幅と高さを計算
        var width = boundsBefore[2].value - leftBefore;
        var height = boundsBefore[3].value - topBefore;

        // 画像が配置されている範囲を切り抜く
        app.activeDocument.crop([leftBefore, topBefore, leftBefore + width, topBefore + height]);

        // 新しい座標を取得
        var boundsAfter = app.activeDocument.activeLayer.bounds;
        var leftAfter = boundsAfter[0].value;
        var topAfter = boundsAfter[1].value;

        // レイヤー名を取得
        var layerName = layer.name.replace(/[\\\/\:\*\?\"\<\>\|]/g, "_"); // 予期しない文字を置換

        // 画像を出力
        var outputFile = new File(outputPath + "/" + layerName + "_@x" + leftBefore + "_@y" + topBefore + "_@l" + tempNum +".png");
        var pngSaveOptions = new PNGSaveOptions();
        app.activeDocument.saveAs(outputFile, pngSaveOptions, true);

        tempNum -= 1;

        // 元の状態に戻す
        app.activeDocument.activeHistoryState = app.activeDocument.historyStates[app.activeDocument.historyStates.length - 2];
        layer.visible = false;  // レイヤーを非表示
    }

    // すべてのレイヤーを再表示
    for (var i = 0; i < app.activeDocument.layers.length; i++) {
        app.activeDocument.layers[i].visible = true;
    }

    // 完了メッセージ
    alert("処理が完了しました");
}
