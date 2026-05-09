# ASL Realtime Recognition

A real-time American Sign Language (ASL) recognition system using TensorFlow, OpenCV, and MediaPipe.

## Features

* Real-time webcam hand tracking
* ASL alphabet recognition
* MediaPipe hand landmark detection
* TensorFlow deep learning model
* Interactive on-screen controls
* Word building system
* Prediction stabilization buffer

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* MediaPipe
* NumPy

## Project Structure

```bash
asl-realtime-recognition/
│
├── main.py
├── requirements.txt
├── README.md
├── hand_landmarker.task
└── bestModel_final_v2.keras
```

## Installation

Clone the repository:

```bash
git clone https://github.com/FeridunTavsanli/asl-realtime-recognition.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

## Controls

* Add → Adds the predicted letter
* Delete → Deletes last letter
* Space → Adds space
* Finish → Completes the word

## How It Works

The system uses MediaPipe hand landmark detection to locate the hand in real time. The cropped hand image is processed and passed into a TensorFlow model for ASL letter classification.

Predictions are stabilized using a buffer system to reduce noise and improve recognition accuracy.

## Future Improvements

* Sentence generation
* Speech output
* Improved accuracy
* Dynamic gesture recognition

## License

MIT License
