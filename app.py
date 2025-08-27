import cv2  # OpenCV for video frame processing
from flask import Flask, render_template, Response, send_from_directory  # Flask web framework
from flask_socketio import SocketIO, emit  # For real-time communication with clients
from engage_track_video import EngageTrackVideoProcessor  # Custom video processor for engagement tracking
import time  # For timing and intervals
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
# Initialize SocketIO for real-time communication
socketio = SocketIO(app)
# AssemblyAI API key for speech-to-text (replace with your own key)
API_KEY = os.environ["ASSEMBLY_API_KEY"]
# Create the video processor instance
processor = EngageTrackVideoProcessor(API_KEY)

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

def gen():
    """
    Video frame generator for streaming to the client.
    Emits activity updates and event logs via SocketIO at regular intervals.
    Yields JPEG-encoded frames for the video feed.
    """
    last_activity_emit = 0  # Last time activities were emitted
    activity_emit_interval = 0.1  # Interval (seconds) between activity updates
    while not processor.shutdown_event.is_set():
        frame = processor.get_current_frame()  # Get the latest video frame
        if frame is None:
            continue  # Skip if no frame is available

        current_time = time.time()
        # Emit activities to the client at the specified interval
        if current_time - last_activity_emit >= activity_emit_interval:
            activities = processor.get_activities()
            socketio.emit('update_activities', activities)
            last_activity_emit = current_time

        # Emit new event logs to the client
        new_events = processor.get_new_events()
        for event in new_events:
            socketio.emit('update_log', {
                'timestamp': event[0],
                'event_type': event[1],
                'description': event[2],
                'speech_context': event[3] or ''
            })

        # Encode the frame as JPEG and yield for streaming
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Route for streaming the video feed to the client."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/end_session')
def end_session():
    """Route to end the engagement session and release resources."""
    processor.end_session()
    processor.release()
    return render_template("end.html")


@app.route('/download')
def download():
    """
    Protected route to download the engagement report PDF from the logs directory.
    """
    directory = 'logs'  # Directory containing files to download
    return send_from_directory(directory, "engagement_report.pdf", as_attachment=True)
if __name__ == "__main__":
    # Main entry point: start the Flask app with SocketIO
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    finally:
        # Ensure resources are released on shutdown
        processor.release()