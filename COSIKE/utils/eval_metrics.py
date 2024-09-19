import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support,f1_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import csv
# def eval_affwild(preds, label_orig):
#     val_preds = preds.cpu().detach().numpy()
#     val_true = label_orig.cpu().detach().numpy() 
#     predicted_label = []
#     true_label = []
#     for i in range(val_preds.shape[0]):
#         predicted_label.append(np.argmax(val_preds[i,:],axis=0) ) #
#         true_label.append(val_true[i])
#     macro_av_f1 = f1_score(true_label, predicted_label, average='macro')
#     return macro_av_f1


def eval_meld(results, truths, test=False):
    test_preds = results.cpu().detach().numpy()   #（num_utterance, num_label)
    test_truth = truths.cpu().detach().numpy()  #（num_utterance）
    predicted_label = []
    true_label = []
    for i in range(test_preds.shape[0]):
        predicted_label.append(np.argmax(test_preds[i,:],axis=0) ) #
        true_label.append(test_truth[i])

    if test:
        accuracy = accuracy_score(true_label, predicted_label)
        recall = recall_score(true_label, predicted_label, average='weighted')
        precision = precision_score(true_label, predicted_label, average='weighted')
        wg_av_f1 = f1_score(true_label, predicted_label, average='weighted')
        f1_each_label = f1_score(true_label, predicted_label, average=None)
        print('**TEST** | f1 on each class (Neutral, Surprise, Fear, Sadness, Joy, Disgust, Anger): \n', f1_each_label)
        print('accuracy,recall,precision,wg_av_f1:',accuracy,recall,precision,wg_av_f1)
        # 转换为NumPy数组
        predicted_labels = np.array(predicted_label)
        true_labels = np.array(true_label)
        # 写入CSV文件
        csv_filename = 'FacialMMT_meld_wo_jubu.csv'
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Predicted Label', 'True Label'])  # 写入表头
            for pred, true in zip(predicted_labels, true_labels):
                writer.writerow([pred, true])
        print(f"Predicted and true labels have been written to {csv_filename}")
    else:
        accuracy = accuracy_score(true_label, predicted_label)
        recall = recall_score(true_label, predicted_label, average='weighted')
        precision = precision_score(true_label, predicted_label, average='weighted')
        wg_av_f1 = f1_score(true_label, predicted_label, average='weighted')
        print('accuracy,recall,precision,wg_av_f1:', accuracy, recall, precision, wg_av_f1)
    return wg_av_f1

