import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)  # 14*2 coz there are 14 keypoints and each keypoint has x and y coordinates
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # [3, 224, 224]
        image_tensor = self.transform(img_rgb).unsqueeze(0)  # coz of unsqueeze [1, 3, 224, 224]

        with torch.no_grad():
            outputs = self.model(image_tensor)

        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = img_rgb.shape[:2]

        # the model returns resized image of 224*224, so we need to convert it back to 240*240 (original)
        # x or y --> 224
        # og_x or og_y --> original_w or original_h
        # therefore, og_x or og_y = (x or y) * (original_w or original_h) / 224
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        return keypoints

    # draw single keypoint
    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])

            cv2.putText(image, str(i // 2), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # 5 is the radius; -1 means circle will be filled

        return image

    # draw multiple keypoints
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)

        return output_video_frames