import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self) -> object:
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device('cpu')
model = ConvNet()
model.load_state_dict(torch.load("C:\\Users\\riyad\\Downloads\\model1.pth", map_location=device))

classes=('airplane','ant','banana','baseball','bird','bucket','butterfly','cat','coffee cup','dolphin','donut','duck','fish','leaf','mountain','pencil','smiley face','snake','umbrella','wine bottle')


drawing = False  # true if mouse is pressed
pt1_x, pt1_y = None, None
l = 0


# mouse callback function
def line_drawing(event, x, y, flags, param):
    global pt1_x, pt1_y, drawing, l

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=9)
            pt1_x, pt1_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=9)
        l = 27


img = np.zeros((600, 600, 3), np.uint8)
cv2.namedWindow('test draw')
cv2.setMouseCallback('test draw', line_drawing)

while (1):
    cv2.imshow('test draw', img)
    p = cv2.waitKey(1) & 0xFF

    if p == ord('q'):
        break

    if p==ord('a'):
        img=np.zeros((600,600,3), np.uint8)

    if p == 32:
        img = np.zeros((600, 600, 3), np.uint8)
    if l == 27:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edged=cv2.Canny(gray,0,250)
        (cnts, _)=cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        x,y,w,h=cv2.boundingRect(cnts[-1])
        new_img=gray[y:y+h,x:x+w]

        input_img = cv2.resize(new_img, (28, 28), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(input_img)
        x = x.reshape(1, 1, 28, 28)

        res = model(x.float())
        i = torch.argmax(res)

        print("I think it's a ", classes[i.item()])
        l = 0

cv2.destroyAllWindows()