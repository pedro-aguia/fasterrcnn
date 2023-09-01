import os
import torch
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def frcnn_formater(preds: torch.Tensor, box_thresh: float = 0.5) -> pd.DataFrame:
    data = np.zeros((len(preds[0]["boxes"]), 6))
    bboxes = preds[0]["boxes"].tolist()
    labels = preds[0]["labels"].tolist()
    confidences = preds[0]["scores"].tolist()

    for i in range(len(bboxes)):
        if confidences[i] > box_thresh:
            data[i] = [bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3], confidences[i], labels[i] - 1]

    columns = ["xmin", "ymin", "xmax", "ymax", "confidence", "class"]
    df = pd.DataFrame(data, columns=columns)
    return df.loc[~(df == 0).all(axis=1)]

def plot_confusion_matrix(y_true, y_pred, classes, path,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.xlim(-0.5, len(np.unique(y_true))-0.5)
    plt.ylim(len(np.unique(y_true))-0.5, -0.5)
    plt.savefig(path)

def create_frcnn():
    """Crea un modelo de detección RCNN y lo devuelve"""
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 6)
    return model

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--model_path', type = str)
    parser.add_argument('--imgs_path', type=str)
    parser.add_argument('--export_path', type=str)
    parser.add_argument('--conf_thresh', type=float, default=0.5)
    parser.add_argument('--iou_thresh', type=float, default=0.5)
    parser.add_argument('--output_path', type=str)

    return parser

def predecir(model_path, imgs_path, export_path, conf_thresh, output_path, iou_thresh):
    clave_proyecto = "clcusoyxl0mic070f2k7p938p"

    img_list = os.listdir(imgs_path)
    f = open(export_path)
    data_imgs = json.load(f)
    clases_reales = []
    clases_predecidas = []

    print("Cargando Faster R-CNN...")
    models = {}
    models["frcnn"] = create_frcnn()
    if torch.cuda.is_available():
        models["frcnn"].load_state_dict(torch.load(os.path.join(model_path))["model_state_dict"])
        models["frcnn"].cuda()
    else:
        models["frcnn"].load_state_dict(
            torch.load(os.path.join(models_path, model_name), map_location=torch.device("cpu"))[
                "model_state_dict"
            ]
        )
    models["frcnn"].eval()

    names = ['Bac','GrB','GrL','GrT','PCo_Low-random',
                               'PCo_Low-longitudinal', 'PCo_Medium',
                               'PCo_High', 'PCo_X-uncertain']
    detecciones = []

    for fn in img_list:
        if fn[-3:] == 'xml':
            continue

        labeled_img = next(item for item in labeled_data if item['data_row']['external_id'] == filename)
        aux_labels = labeled_img['projects'][clave_proyecto]['labels'][0]['annotations']['objects']
        labels = []
        nombres = []
        for label in aux_labels:
            if label['name'] not in ['Lin', 'Cap']:
                if len(label['classifications']) > 0:
                    nombre = label['name'] + '_' + label['classifications'][0]['radio_answer']['name']
                else:
                    nombre = label['name']
                labels.append(label)
                nombres.append(nombre)
        xmin = [x["bounding_box"]["left"] for x in labels]
        xmax = [x["bounding_box"]["left"] + x["bounding_box"]["width"] for x in labels]
        ymin = [x["bounding_box"]["top"] for x in labels]
        ymax = [x["bounding_box"]["top"] + x["bounding_box"]["height"] for x in labels]
        #ymax = [h - x["bbox"]["top"] for x in labels]
        #ymin = [h - x["bbox"]["top"] - x["bbox"]["height"] for x in labels]
        bboxes_labeled = [[xmin[i], ymin[i], xmax[i], ymax[i]] for i in range(len(xmin))]
        #categories_labeled = [x["title"] for x in labels]
        categories_labeled = nombres
        indices_aux = list(np.arange(len(categories_labeled)))
        
        img = (cv2.imread(os.path.join(imgs_path, fn)), fn)
        _, extension = os.path.splitext(fn)
        results_filename = img[1].replace("_sub" + extension, "_inf.txt")
        result_files.append(results_filename)

        for name, model in models.items():
            if name == "frcnn":
                with torch.no_grad():
                    trans = transforms.Compose(
                        [
                            lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB),
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                        ]
                    )
                    if torch.cuda.is_available():
                        cuda_img = trans(img[0]).unsqueeze(0).cuda()
                        results = frcnn_formater(model(cuda_img))
                    else:
                        cpu_img = trans(img[0]).unsqueeze(0)
                        results = frcnn_formater(model(cpu_img))
                    results[["xmin", "ymin", "xmax", "ymax", "class"]] = (
                        results[["xmin", "ymin", "xmax", "ymax", "class"]].round().astype(int)
                    )
                    results["confidence"] = results["confidence"].round(2)
                    results.insert(0, "model", "frcnn")

        keep_results = results[results.confiedence > conf_thresh]
        
        for i in range(len(keep_results)):
            dictio = {
                "file": fn,
                "model": "frcnn",
                "xmin": max(keep_results.iloc[i, 'xmin'], 0),
                "ymin": max(keep_results.iloc[i, 'ymin'], 0),
                "xmax": keep_results.iloc[i, 'xmax'],
                "ymax": keep_results.iloc[i, 'ymax'],
                "confidence": keep_results.iloc[i, 'confidence'],
                "class": keep_results.iloc[i, 'class'],
                "name": names[keep_results.iloc[i, 'class']]
                }
            detecciones.append(dictio)

            xx1 = np.maximum(xmin, dictio['xmin'])
            yy1 = np.maximum(ymin, dictio['ymin'])
            xx2 = np.minimum(xmax, dictio['xmax'])
            yy2 = np.minimum(ymax, dictio['ymax'])
            w2 = np.maximum(0, xx2 - xx1 + 0.000001)
            h2 = np.maximum(0, yy2 - yy1 + 0.000001)
            intersections = np.array(w2) * np.array(h2)
            union1 = (np.array(xmax) - np.array(xmin)) * (np.array(ymax) - np.array(ymin))
            union2 = (dictio['xmax'] - dictio['xmin']) * (
                dictio['ymax'] - dictio['ymin'])
            unions = union1 - intersections + union2
            iou = list(intersections / unions)
            if len(iou) == 0:
                clases_reales.append(10)
                clases_predecidas.append(dictio['class'])
            else:
                indice_aux = iou.index(max(iou))


                clases_predecidas.append(dictio['class'])
                if iou[indice_aux] > iou_thresh:
                    clases_reales.append(numbers[names.index(
                        categories_labeled[indice_aux])])
                    
                    if indice_aux in indices_aux:
                        indices_aux.remove(indice_aux)
                    
                else:
                    clases_reales.append(10)
                    
        for indice_aux in indices_aux:
            clases_predecidas.append(10)
            clases_reales.append(numbers[names.index(categories_labeled[indice_aux])])

    myFile = open(os.path.join(output_path, 'anotations.csv'), 'w', newline = '')
    writer = csv.writer(myFile, delimiter = ';')
    writer.writerow(list(detecciones[0].keys()))
    for dictionary in detecciones:
        writer.writerow(dictionary.values())
    myFile.close()

    if json_path != '':
        conf_out = os.path.join(output_path, 'detr_conf.png')
        for k in clases_reales:
            if isinstance(k, str):
                print(k, type(k))
        for l in clases_predecidas:
            if isinstance(l, str):
                print(l, type(l))
        plot_confusion_matrix(clases_reales, clases_predecidas, path = conf_out, normalize=True,
                              classes = ['Bac','GrB','GrL','GrT','PCo_Low-random',
                       'PCo_Low-longitudinal', 'PCo_Medium',
                       'PCo_High', 'PCo_X-uncertain', 'background'], title='Normalized confusion matrix')

if __name__ == "__main__":
    parser = argparse.ArgumentParser('FRCNN training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    predecir(args.model_path, args.imgs_path, args.export_path,
             args.conf_thresh, args.output_path, args.iou_thresh)
