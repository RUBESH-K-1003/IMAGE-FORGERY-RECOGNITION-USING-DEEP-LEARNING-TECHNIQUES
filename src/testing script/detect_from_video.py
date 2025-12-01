"""
Run DeepFake detection on a video using Xception.
This version has hardcoded paths for model, input video, and output folder.
"""

import os
import cv2
import dlib
import torch
from PIL import Image as pil_image
from tqdm import tqdm

# --- Hardcoded paths ---
MODEL_PATH = r"C:\Users\RUBESH\Desktop\RK Rough\Sem-7\velu model\best_model_fast.pth"      # your trained model
VIDEO_PATH = r"C:\Users\RUBESH\Downloads\fakeandoriginalvideos\Deepfakes.mp4"  # input video
OUTPUT_PATH = r"C:\Users\RUBESH\Desktop\RK Rough\Sem-7\velu model\Testing interface\output"                   # folder to save annotated video


# ======================
# Xception Architecture
# (imported version simplified to match your checkpoint)
# ======================
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,
                               padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x+=skip
        return x


class Xception(nn.Module):
    def __init__(self, num_classes=1):
        super(Xception, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1)
        self.block5=Block(728,728,3,1)
        self.block6=Block(728,728,3,1)
        self.block7=Block(728,728,3,1)
        self.block8=Block(728,728,3,1)
        self.block9=Block(728,728,3,1)
        self.block10=Block(728,728,3,1)
        self.block11=Block(728,728,3,1)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)
        self.last_linear = self.fc

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x


# ======================
# Transforms
# ======================
from torchvision import transforms
xception_default_data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(333),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
}


# ======================
# Helper: Face bounding box
# ======================
def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, size_bb


# ======================
# Main Function
# ======================
def run_inference():
    print(f"Starting: {VIDEO_PATH}")

    reader = cv2.VideoCapture(VIDEO_PATH)
    if not reader.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    video_fn = os.path.splitext(os.path.basename(VIDEO_PATH))[0] + '_out.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(os.path.join(OUTPUT_PATH, video_fn), fourcc, fps, (width, height))

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    # Load model
    print("[INFO] Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    model = Xception(num_classes=1)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("[INFO] Model loaded!")

    pbar = tqdm(total=int(reader.get(cv2.CAP_PROP_FRAME_COUNT)))
    while reader.isOpened():
        ret, image = reader.read()
        if not ret:
            break

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)

        if len(faces):
            face = sorted(faces, key=lambda x: (x.right()-x.left())*(x.bottom()-x.top()), reverse=True)[0]
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y+size, x:x+size]

            with torch.no_grad():
                preprocessed_face = xception_default_data_transforms['test'](pil_image.fromarray(cropped_face))
                tensor = preprocessed_face.unsqueeze(0).to(device)
                output = model(tensor)
                prediction = torch.sigmoid(output).item()

            face_prediction = 'FAKE' if prediction > 0.5 else 'REAL'

            confidence = prediction if face_prediction == 'FAKE' else 1 - prediction
            color = (0, 0, 255) if face_prediction == 'FAKE' else (0, 255, 0)
            cv2.rectangle(image, (x, y), (x + size, y + size), color, 2)
            text = f"{face_prediction}: {confidence:.2f}"
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        writer.write(image)
        pbar.update(1)

    reader.release()
    writer.release()
    pbar.close()
    print(f"Finished! Output saved to {os.path.join(OUTPUT_PATH, video_fn)}")


if __name__ == "__main__":
    run_inference()
