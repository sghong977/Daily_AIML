from ensemble_boxes import *
import json

""" For using ensemble algorithms ***
INPUT: boxes_list, scores_list, labels_list

1. boxes_list[model][bbox][x1, y1, x2, y2]
- bbox information should be normalized in range [0; 1]

2. scores_list[model][score]

3. labels_list[model][score]
"""
image_w, image_h = 1000, 1000 #640, 480

# CONSENSUS
def preprocessing(pred_outputs):
    pass

    return pred_outputs

# loading(): convert json files into dict{}
# dict['image_id'] = {'bbox', 'score', 'category_id'}
# each 'bbox', 'score', 'category_id' is stacked with different model outputs
def loading(jsons):
    global image_w, image_h
    # first dim: image
    pred_outputs = dict()
    boxes_list, scores_list, labels_list = [],[],[]

    # 1 JSON OUTPUT: list of dict{'image_id', 'bbox', 'score', 'category_id']}
    for j in range(len(jsons)):
        image_id = 0
        with open(jsons[j], 'r') as file:
            data = json.load(file)
            tmp_box, tmp_s, tmp_l = [],[],[]
            for im in data:
                if str(image_id) != str(im['image_id']):
                    if j == 0:
                        pred_outputs[image_id] = {'bbox':[tmp_box], 'score':[tmp_s], 'category_id':[tmp_l]}
                    else:
                        pred_outputs[image_id]['bbox'].append(tmp_box)
                        pred_outputs[image_id]['score'].append(tmp_s)
                        pred_outputs[image_id]['category_id'].append(tmp_l)
                    image_id += 1
                    tmp_box, tmp_s, tmp_l = [],[],[]
                x1, y1, w, h = im['bbox']
                x2 = x1 + w
                y2 = y1 + h
                box = [x1/image_w, y1/image_h, x2/image_w, y2/image_h]
                for b in range(len(box)):
                    if box[b] < 0: box[b]=0.0
                    elif box[b] > 1: box[b]=1.0

                # normalize and minus?
                tmp_box.append(box)
                tmp_s += [im['score']]
                tmp_l += [im['category_id']]
            # save final image
            if j == 0:
                pred_outputs[image_id] = {'bbox':[tmp_box], 'score':[tmp_s], 'category_id':[tmp_l]}
            else:
                pred_outputs[image_id]['bbox'].append(tmp_box)
                pred_outputs[image_id]['score'].append(tmp_s)
                pred_outputs[image_id]['category_id'].append(tmp_l)
    
    return pred_outputs


def main(jsons, method='wbf', preprocessing=False, out_path='./ensemble_result2.json'):
    global image_w, image_h

    # parameters
    iou_thr = 0.5
    skip_box_thr = 0.01
    sigma = 0.1
    weights = [1] * len(jsons) #model 가중치

    # load
    pred_outputs = loading(jsons)

    # pre-procesing (consensus)
    if preprocessing:
        pred_outputs = preprocessing(pred_outputs)

    # per-image ensemble result
    json_data=[]
    for im in range(len(pred_outputs)):
        boxes_list, scores_list, labels_list = pred_outputs[im]['bbox'], pred_outputs[im]['score'], pred_outputs[im]['category_id']

        # Apply Algorithm
        if method=='nms':
            boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
        elif method == 'soft_nms':
            boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
        elif method == 'wnms':
            boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        elif method == 'wbf':
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        else:
            print("INVALID METHOD")
        #print(len(boxes), len(boxes_list[0]), len(boxes_list[1]), len(boxes_list[2]))

        # save per bbox (in one image)
        for j in range(len(boxes)):
            # denormalize
            boxes[j][0]=boxes[j][0]*image_w
            boxes[j][1]=boxes[j][1]*image_h
            boxes[j][2]=boxes[j][2]*image_w - boxes[j][0]
            boxes[j][3]=boxes[j][3]*image_h - boxes[j][1]

            results = {'image_id':im, "bbox": boxes[j].tolist(), "score":scores[j],"category_id":int(labels[j])}
            # print(results)
            json_data.append(results)
            
    # write json file
    submission_ensemble = open(out_path,'w',encoding='utf-8')
    submission_ensemble.write(json.dumps(json_data))


""" ENSEMBLE!
INPUT
1. json list (좋은 것들 다다익선?!)
2. method (기본 'wbf'로 해뒀습니다!)
3. output json_path와 세부 파라미터 (iou threshold 등)는 main()함수 내에서 수정 부탁드립니다!

OUTPUT
json 제출파일
main()에 전달, 기본으로 out_path='./ensemble_result2.json'로 해뒀습니다.
"""
if __name__ == '__main__':
    jsons = [
        "/raid/sghong/Detection/mmdetection/work_dirs/yoloxx_0826/inferences/bbox.json", #28
        #"/nas4/viprior/sbhong/yoloxx/mmcls/submission.json",                             #28
        "/raid/sghong/Detection/mmdetection/work_dirs/yoloxx/inferences/bbox.json",
        "/raid/sghong/Detection/mmdetection/work_dirs/yolox_cls2/inferences/bbbox.json", # 성능 까먹음. 아마 구릴 것
    ]
    main(jsons, 'wbf')
