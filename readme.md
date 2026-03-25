## Engagement Tracker
An AI-powered real-time engagement tracker that analyzes video feeds, speech activity, and user interactions. It leverages Flask and Socket.IO for a dynamic web interface, OpenCV for video processing, and MediaPipe (via a custom processor) for detecting activities and engagement levels. The system provides live dashboards and generates downloadable PDF engagement reports at the end of each session.

![Demo](https://raw.githubusercontent.com/Joshluk3328j/Engagment-tracker-v1/main/img.png)

## 🚀 Features

- 📹 Real-time video streaming using OpenCV and Flask. <br>
- 🧠 Engagement tracking with Media Pipe for face, gaze, and activity detection. <br>
- 🎙️ Speech-to-text integration powered by the AssemblyAI API. <br>
- 🔄 Live activity updates and event logs via Socket.IO. <br>
- 📑 Session reports exported as downloadable PDFs. <br>
- 🌐 Web-based dashboard for monitoring participants in real time. <br>

## 🛠️ Installation
Follow these steps to set up the Engagement Tracker on your local machine.
1. Clone the Repository
```
git clone https://github.com/yourusername/engagement-tracker.git
cd engagement-tracker
```
2. Create a Virtual Environment
```
python3 -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```
3. Install Dependencies

```
pip install -r requirements.txt
```
4. Set Environment Variables
Create a .env file in the project root and add your AssemblyAI API key:
```
ASSEMBLY_API_KEY=your_api_key_here
```

## ▶️ Running the App
Start the Flask application:
```
python app.py
```
The server will run by default at:👉 http://localhost:5000
## 📂 Project Structure
Engagement Tracker/
```
├── app.py                    # Main Flask app entry point
├── engage_track_video.py     # Engagement tracking video processor
├── templates/
│   ├── index.html            # Dashboard UI
│   └── end.html              # End-session page
├── logs/
│   └── engagement_report.pdf # Session reports stored here
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (AssemblyAI key)
```
## 📊 Workflow

- Video Capture: Video frames are captured using OpenCV. <br>
- Engagement Detection: Frames are processed by EngageTrackVideoProcessor for face, gaze, and activity detection. <br>
- Real-time Updates: Socket.IO emits activities and events to the client dashboard. <br>
- Session Logging: Session logs are written to disk. <br>
- Report GeneraEngagement Tracker/tion: Engagement reports are generated as downloadable PDFs. <br>

## 📦 Dependencies

|Dependency |Description|
|-----------|-----------|
|Flask|Web framework for the application.|
|Flask-SocketIO|Real-time communication for live updates.|
|OpenCV (cv2)|Video capture and processing.|
|MediaPipe|Framework for engagement detection.|
|python-dotenv|Environment variable management.|
|AssemblyAI|API for speech-to-text functionality.|

## 📜 License
This project is licensed under the MIT License. Feel free to use and modify it for your own projects.
