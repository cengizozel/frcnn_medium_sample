import torch
import cv2
import os
from utils import (
    get_model_instance_segmentation
)

def draw_bbox(box, img):
    cv2.rectangle(img, box[0], box[1], color=(0, 255, 0), thickness=3)

def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score)<threshold:
        return []
    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        person_boxes.append(box)

    return person_boxes


def run_detection(image_bgr, file, save_folder, is_video, out=None, frame=0, length=0):
    if is_video:
        save_path = f"my_data/results/{save_folder}/{file}"
    else:
        save_path = f"my_data/results/{save_folder}/{file}"

    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().to(CTX)
    input = torch.stack([img_tensor])

    box = get_person_detection_boxes(box_model, input, 0.95)
    if len(box) > 0:
        for b in box:
            draw_bbox(b, image_bgr)
        if is_video:
            out.write(image_bgr)
            if frame < length-1:
                print(f'Writing frame {frame}/{length-1}', end='\r')
            else:
                print(f'Writing frame {frame}/{length-1}\nVideo saved to {save_path}\n')
        else:
            cv2.imwrite(save_path, image_bgr)
            print(f"\nFound hand at {box}")
            print(f"Result image has been saved as {save_path}")
    else:
        if is_video:
            out.write(image_bgr)
            if frame < length-1:
                print(f'Writing frame {frame}/{length-1}', end='\r')
            else:
                print(f'Writing frame {frame}/{length-1}\nVideo saved to {save_path}\n')
        else:
            print(f"No hand detected at {file}")
            os.remove(save_path)


CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

box_model = get_model_instance_segmentation(2)
box_model.load_state_dict(torch.load('models/tuned_hand_blur.pth'), strict=False)
box_model.to(CTX)
box_model.eval()

# print(box_model)
# print(box_model.state_dict())


# Detect hand in image
def detect_img():
    test_path = 'my_data/test/lab3'
    save_folder = 'lab3'
    for i, file in enumerate(os.listdir(test_path)):
        image_bgr = cv2.imread(f'{test_path}/{file}')
        run_detection(image_bgr, file, save_folder, False)


# Detect hand in mp4 video
def detect_video():
    save_folder = 'video'
    file = 'lab_video_0.mp4'
    video_path = f'my_data/test/video/{file}'
    vidcap = cv2.VideoCapture(video_path)

    save_path = f"my_data/results/{save_folder}/{file}"
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(save_path, fourcc, 30, (width, height), isColor=True)
    
    frame = 0
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print()
    while vidcap.isOpened():
        ret, image_bgr = vidcap.read()
        if ret:
            run_detection(image_bgr, file, save_folder, True, out, frame, length)
            frame += 1
            # cv2.imshow('frame', image_bgr)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    vidcap.release()
    cv2.destroyAllWindows()


# Switch between image and video detection when running script
# File name parameters can be changed directly inside the functions
detect_video()
