'''
YOLO 결과 화면 구현 
'''
import argparse
import gradio as gr
import cv2
import os
import time 
import ffmpeg
from ultralytics import YOLO
import sys
import io
import contextlib
import numpy as np
import pandas as pd
from datetime import datetime

'''
object_name 변수
 : 맨 밑에 makdowon과 Model에서 사용합니다. 
'''
object_name = "Fashion AI Caster"
log_file = 'output.log'


class Logger:
    def __init__(self, filename, stream, max_lines=20):
        self.stream = stream
        self.filename = filename
        self.max_lines = max_lines
        self.log = open(filename, "r+")
        self.keep_last_lines()

    def keep_last_lines(self):
        self.log.seek(0)
        lines = self.log.readlines()
        if len(lines) > self.max_lines:
            self.log.seek(0)
            self.log.truncate()
            self.log.writelines(lines[-self.max_lines:])

    def write(self, message):
        self.stream.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure all writes are flushed to the file before reading
        self.keep_last_lines()  # Check and keep only the last max_lines after each write

    def flush(self):
        self.stream.flush()
        self.log.flush()
        
    def isatty(self):
        return False

sys.stdout = Logger("output.log", sys.stdout)
sys.stderr = Logger("output.log", sys.stderr)

def read_logs():
    sys.stdout.flush()
    with open("output.log", "r") as f:
        return f.read()


## yolo 실행
def run_yolo(input_video_url):
    # YOLO 모델 지정 
    model = YOLO(f'./weights/' + os.listdir('./weights/')[0])

    cap = cv2.VideoCapture(input_video_url)
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    video_name = os.path.basename(input_video_url)
    output_path = os.getcwd()+'/video/out/' + video_name
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    df = pd.DataFrame(columns=['time', 'object'])

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        result = model(frame)

        if len(result[0].boxes.cls) > 0:
            now = datetime.now()
            print(now)

            for i in range(len(result[0].boxes.cls)):
                obj_name = result[0].names[np.array(result[0].boxes.cls)[i]]
                print(obj_name)
                new_row = pd.DataFrame({'time': [now], 'object': [obj_name]})
                df = pd.concat([df, new_row], ignore_index=True)

        img = result[0].plot()
        out.write(img)

    df.to_csv('detect_log_test.csv', index=False)
    cap.release()
    out.release()
    resized_video = 'video/resized_/' + time.strftime('%Y%m%d%H%M') + '.mp4'
    ffmpeg.input(output_path).output(resized_video, crf=35).run() # 영상 용량 축소
    return input_video_url, resized_video


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--server_name',
        type=str,
        default='0.0.0.0'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=7860
    )
    args=parser.parse_args()

    # Gradio UI
    with gr.Blocks() as demo:

        with gr.Row():

            with gr.Column():
                markdown = gr.Markdown(f"# {object_name}")
                input1 = gr.Textbox(label = "Video URL", value="http://evc.re.kr:20096/www/test_data/v4_demo1.mp4") # Video URL 넣기 
                btn1 = gr.Button("Run", size="sm")

            with gr.Column():
                output1 = gr.Video(autoplay=True) # 원본 비디오 재생

            with gr.Column():
                output2 = gr.Video(autoplay=True) # 결과 비디오 재생
             
            btn1.click(fn=run_yolo, inputs=input1, outputs=[output1, output2])

        logs = gr.Textbox()
        demo.load(read_logs, None, logs, every=1)

    demo.queue().launch(
            server_name=args.server_name,
            server_port=args.server_port,
            debug=True
        )