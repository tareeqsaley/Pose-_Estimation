{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0e4cfd40",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Defaulting to user installation because normal site-packages is not writeable\n",
      "Requirement already satisfied: mediapipe in /usr/local/lib/python3.6/dist-packages (0.8.5-cuda102)\n",
      "Collecting opencv-python\n",
      "  Downloading opencv_python-4.5.5.64-cp36-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl (39.2 MB)\n",
      "     |████████████████████████████████| 39.2 MB 68 kB/s              \n",
      "\u001b[?25hRequirement already satisfied: pandas in /usr/lib/python3/dist-packages (0.22.0)\n",
      "Collecting scikit-learn\n",
      "  Downloading scikit_learn-0.24.2-cp36-cp36m-manylinux2014_aarch64.whl (24.0 MB)\n",
      "     |████████████████████████████████| 24.0 MB 67 kB/s              \n",
      "\u001b[?25hRequirement already satisfied: protobuf>=3.11.4 in /usr/local/lib/python3.6/dist-packages (from mediapipe) (3.19.4)\n",
      "Requirement already satisfied: wheel in /usr/lib/python3/dist-packages (from mediapipe) (0.30.0)\n",
      "Requirement already satisfied: six in /usr/lib/python3/dist-packages (from mediapipe) (1.11.0)\n",
      "Requirement already satisfied: numpy==1.19.4 in /usr/local/lib/python3.6/dist-packages (from mediapipe) (1.19.4)\n",
      "Requirement already satisfied: absl-py in /usr/local/lib/python3.6/dist-packages (from mediapipe) (1.0.0)\n",
      "Requirement already satisfied: opencv-contrib-python in /usr/local/lib/python3.6/dist-packages (from mediapipe) (4.5.5.64)\n",
      "Requirement already satisfied: attrs>=19.1.0 in /home/tareeq/.local/lib/python3.6/site-packages (from mediapipe) (21.4.0)\n",
      "Collecting threadpoolctl>=2.0.0\n",
      "  Downloading threadpoolctl-3.1.0-py3-none-any.whl (14 kB)\n",
      "Requirement already satisfied: scipy>=0.19.1 in /usr/lib/python3/dist-packages (from scikit-learn) (0.19.1)\n",
      "Collecting joblib>=0.11\n",
      "  Downloading joblib-1.1.0-py2.py3-none-any.whl (306 kB)\n",
      "     |████████████████████████████████| 306 kB 23.5 MB/s            \n",
      "\u001b[?25hInstalling collected packages: threadpoolctl, joblib, scikit-learn, opencv-python\n",
      "Successfully installed joblib-1.1.0 opencv-python-4.5.5.64 scikit-learn-0.24.2 threadpoolctl-3.1.0\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install mediapipe opencv-python pandas scikit-learn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "ba4fd0d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import mediapipe as mp\n",
    "import cv2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "3e133e48",
   "metadata": {},
   "outputs": [],
   "source": [
    "mp_drawing = mp.solutions.drawing_utils # Drawing helpers\n",
    "mp_holistic = mp.solutions.holistic # Mediapipe Solutions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "3911eda1",
   "metadata": {},
   "outputs": [],
   "source": [
    "cap = cv2.VideoCapture(0)\n",
    "# Initiate holistic model\n",
    "with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:\n",
    "    \n",
    "    while cap.isOpened():\n",
    "        ret, frame = cap.read()\n",
    "        \n",
    "        # Recolor Feed\n",
    "        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "        image.flags.writeable = False        \n",
    "        \n",
    "        # Make Detections\n",
    "        results = holistic.process(image)\n",
    "        # print(results.face_landmarks)\n",
    "        \n",
    "        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks\n",
    "        \n",
    "        # Recolor image back to BGR for rendering\n",
    "        image.flags.writeable = True   \n",
    "        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)\n",
    "        \n",
    "        # 1. Draw face landmarks\n",
    "        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, \n",
    "                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),\n",
    "                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)\n",
    "                                 )\n",
    "        \n",
    "        # 2. Right hand\n",
    "        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, \n",
    "                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),\n",
    "                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)\n",
    "                                 )\n",
    "\n",
    "        # 3. Left Hand\n",
    "        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, \n",
    "                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),\n",
    "                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)\n",
    "                                 )\n",
    "\n",
    "        # 4. Pose Detections\n",
    "        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, \n",
    "                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),\n",
    "                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)\n",
    "                                 )\n",
    "                        \n",
    "        cv2.imshow('Raw Webcam Feed', image)\n",
    "\n",
    "        if cv2.waitKey(10) & 0xFF == ord('q'):\n",
    "            break\n",
    "\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1e25af98",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
