// �_�C�A���O�ŏo�͐�̃t�H���_��I��
var outputPath = Folder.selectDialog("�o�͐�t�H���_��I��");

var tempNum = app.activeDocument.layers.length;

// ���[�U�[���L�����Z���{�^�����������ꍇ�͏����𒆒f
if (!outputPath) {
    alert("�L�����Z������܂����B�����𒆎~���܂��B");
} else {
    // ���ׂẴ��C���[���\���ɂ���
    for (var i = 0; i < app.activeDocument.layers.length; i++) {
        app.activeDocument.layers[i].visible = false;
    }

    // ���C���[���Ƃɏ���
    for (var i = 0; i < app.activeDocument.layers.length; i++) {
        var layer = app.activeDocument.layers[i];
        layer.visible = true;  // ���C���[��\��

        // �摜���z�u����Ă���͈͂��擾
        var boundsBefore = layer.bounds;
        var leftBefore = boundsBefore[0].value;
        var topBefore = boundsBefore[1].value;

        // �摜���z�u����Ă���͈͂̕��ƍ������v�Z
        var width = boundsBefore[2].value - leftBefore;
        var height = boundsBefore[3].value - topBefore;

        // �摜���z�u����Ă���͈͂�؂蔲��
        app.activeDocument.crop([leftBefore, topBefore, leftBefore + width, topBefore + height]);

        // �V�������W���擾
        var boundsAfter = app.activeDocument.activeLayer.bounds;
        var leftAfter = boundsAfter[0].value;
        var topAfter = boundsAfter[1].value;

        // ���C���[�����擾
        var layerName = layer.name.replace(/[\\\/\:\*\?\"\<\>\|]/g, "_"); // �\�����Ȃ�������u��

        // �摜���o��
        var outputFile = new File(outputPath + "/" + layerName + "_@x" + leftBefore + "_@y" + topBefore + "_@l" + tempNum +".png");
        var pngSaveOptions = new PNGSaveOptions();
        app.activeDocument.saveAs(outputFile, pngSaveOptions, true);

        tempNum -= 1;

        // ���̏�Ԃɖ߂�
        app.activeDocument.activeHistoryState = app.activeDocument.historyStates[app.activeDocument.historyStates.length - 2];
        layer.visible = false;  // ���C���[���\��
    }

    // ���ׂẴ��C���[���ĕ\��
    for (var i = 0; i < app.activeDocument.layers.length; i++) {
        app.activeDocument.layers[i].visible = true;
    }

    // �������b�Z�[�W
    alert("�������������܂���");
}
