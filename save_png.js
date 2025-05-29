//target photoshop

// 源文件夹，包含 PSD 文件
var sourceFolder = new Folder("D:/BaiduSyncdisk/SZY/Essay/dataset/99psFile");
// 目标文件夹，用于保存 PNG 文件
var targetFolder = new Folder("D:/BaiduSyncdisk/SZY/Essay/dataset/03color");

// 检查目标文件夹是否存在，如果不存在则创建
if (!targetFolder.exists) {
    targetFolder.create();
}

// 获取源文件夹中所有 PSD 文件
var psdFiles = sourceFolder.getFiles("*.psd");

// 遍历 PSD 文件
for (var i = 0; i < psdFiles.length; i++) {
    var psdFile = psdFiles[i];

    // 打开 PSD 文件
    var doc = open(psdFile);

    // 设置 PNG 文件的保存路径
    var pngFile = new File(targetFolder + "/" + psdFile.name.replace(".psd", ".png"));

    // 保存为 PNG 格式
    var pngSaveOptions = new PNGSaveOptions();
    doc.saveAs(pngFile, pngSaveOptions, true, Extension.LOWERCASE);

    // 关闭 PSD 文件，不保存更改
    doc.close(SaveOptions.DONOTSAVECHANGES);
}

alert("所有文件已转换并保存为 PNG 格式。");
