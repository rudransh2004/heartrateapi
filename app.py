import cv2
import pyramids
import heartrate
import preprocessing
import eulerian
from flask import Flask,render_template,request
from werkzeug.utils import secure_filename
app = Flask(__name__)  
freq_min = 1
freq_max = 1.8

def give_heart_rate(path):
    print("Reading + preprocessing video...")
    video_frames, frame_ct, fps = preprocessing.read_video(path)
    print("Building Laplacian video pyramid...")
    lap_video = pyramids.build_video_pyramid(video_frames)
    amplified_video_pyramid = []
    for i, video in enumerate(lap_video):
        if i == 0 or i == len(lap_video)-1:
            continue
        print("Running FFT and Eulerian magnification...")
        result, fft, frequencies = eulerian.fft_filter(video, freq_min, freq_max, fps)
        lap_video[i] += result
        print("Calculating heart rate...")
        heart_rate = heartrate.find_heart_rate(fft, frequencies, freq_min, freq_max)

    print("Rebuilding final video...")
    amplified_frames = pyramids.collapse_laplacian_video_pyramid(lap_video, frame_ct)
    print("Heart rate: ", heart_rate, "bpm")
    return heart_rate
@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  

@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file'] 
        f.save(f.filename)  
        print(f)
        x =  give_heart_rate(f.filename)
        return render_template("success.html", name = x) 

if __name__ == '__main__':  
    app.run(debug = False)  

