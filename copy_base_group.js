//target photoshop

// 检查是否有多个打开的文档
if (app.documents.length > 1) {
    // 获取当前活动的文档
    var sourceDoc = app.activeDocument;

    // 让用户选择要复制的组
    var selectedLayer = sourceDoc.activeLayer;

    // 确保选择的是组（图层集）
    if (selectedLayer.typename != "LayerSet") {
        alert("请先选择一个组（图层集）来复制。");
    } else {
        // 获取当前打开的所有文档
        var allDocs = app.documents;

        // 遍历所有文档，除了源文档本身
        for (var i = 0; i < allDocs.length; i++) {
            var targetDoc = allDocs[i];

            if (targetDoc != sourceDoc) {
                // 激活目标文档
                app.activeDocument = targetDoc;

                // 检查图像模式是否为RGB，如果不是则转换为RGB模式
                if (targetDoc.mode != DocumentMode.RGB) {
                    targetDoc.changeMode(ChangeMode.RGB);
                }

                // 复制图层组到目标文档
                app.activeDocument = sourceDoc;  // 切换回源文档
                var duplicatedLayer = selectedLayer.duplicate(targetDoc, ElementPlacement.PLACEATEND);  // 复制组到目标文档

                // 切换回目标文档
                app.activeDocument = targetDoc;

                // 查找"base"组中的"wall"图层并激活
                var baseGroup = null;
                var wallLayer = null;
                for (var j = 0; j < targetDoc.layers.length; j++) {
                    if (targetDoc.layers[j].name == "base" && targetDoc.layers[j].typename == "LayerSet") {
                        baseGroup = targetDoc.layers[j];
                        break;
                    }
                }

                // 如果找到了"base"组，继续查找其中的"wall"图层
                if (baseGroup != null) {
                    for (var k = 0; k < baseGroup.layers.length; k++) {
                        if (baseGroup.layers[k].name == "wall") {
                            wallLayer = baseGroup.layers[k];
                            break;
                        }
                    }

                    // 如果找到了"wall"图层，将其设置为当前激活的图层
                    if (wallLayer != null) {
                        targetDoc.activeLayer = wallLayer;
                    } else {
                        alert("在文档 " + targetDoc.name + " 中的 'base' 组中未找到名为 'wall' 的图层。");
                    }
                } else {
                    alert("在文档 " + targetDoc.name + " 中未找到名为 'base' 的组。");
                }
            }
        }

        // 最后切换回源文档
        app.activeDocument = sourceDoc;

        alert("已将组复制到其他文档，并设置了 'wall' 图层为当前激活的图层。");
    }
} else {
    alert("没有足够的文档打开。请确保至少打开两个文档。");
}
