//target photoshop

// 保存文件的目标文件夹
var saveFolder = new Folder("D:/BaiduSyncdisk/SZY/Essay/dataset/99psFile");

// 检查保存文件夹是否存在，如果不存在则创建
if (!saveFolder.exists) {
    saveFolder.create();
}

// 遍历所有打开的文档
for (var i = app.documents.length - 1; i >= 0; i--) {
    var doc = app.documents[i];
    app.activeDocument = doc;

    // 如果文档名称是 "base.psd"，则跳过
    if (doc.name.toLowerCase() === "base.psd") {
        continue; // 跳过此文件
    }

    // 查找名为 "base" 的组
    var baseGroup = null;
    for (var j = 0; j < doc.layers.length; j++) {
        if (doc.layers[j].name == "base" && doc.layers[j].typename == "LayerSet") {
            baseGroup = doc.layers[j];
            break;
        }
    }

    // 如果找到了 "base" 组
    if (baseGroup != null) {
        // 查找 "base" 组中的 "wall" 图层
        var wallLayer = null;
        for (var k = 0; k < baseGroup.layers.length; k++) {
            if (baseGroup.layers[k].name == "wall") {
                wallLayer = baseGroup.layers[k];
                break;
            }
        }

        // 如果找到了 "wall" 图层
        if (wallLayer != null) {
            // 选择 "wall" 图层
            doc.activeLayer = wallLayer;

            // 创建填充颜色
            var fillColor = new SolidColor();
            fillColor.rgb.red = 210;
            fillColor.rgb.green = 226;
            fillColor.rgb.blue = 237;

            // 对当前选区进行填充
            doc.selection.fill(fillColor, ColorBlendMode.NORMAL, 100, false);

            // 取消选择
            doc.selection.deselect();

            // 设置新保存路径
            var newFilePath = new File(saveFolder + "/" + doc.name.replace(/\.[^\.]+$/, "") + ".psd");

            // 保存当前文档为PSD格式到指定位置
            var psdSaveOptions = new PhotoshopSaveOptions();
            psdSaveOptions.layers = true; // 保留图层信息
            doc.saveAs(newFilePath, psdSaveOptions, true, Extension.LOWERCASE);

            // 关闭当前文档，不保存更改
            doc.close(SaveOptions.DONOTSAVECHANGES);
        } else {
            alert("在文档 " + doc.name + " 中未找到名为 'wall' 的图层。");
        }
    } else {
        alert("在文档 " + doc.name + " 中未找到名为 'base' 的组。");
    }
}

alert("所有文档处理完成，并已保存为PSD文件（跳过了 base.psd）。");
