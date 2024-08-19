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
object_name = "Edge AI Servant"
stop_execution = False # Global flag to control execution

def stop_functions():
    global stop_execution
    stop_execution = True

## stdout logger class
class CaptureOutput(io.StringIO):
    def __init__(self, max_length=100):
        super().__init__()
        self.output = []
        self.max_length = max_length  # 로그를 유지할 최대 줄 수 설정
        self.auto_clear_threshold = 100  # 자동으로 클리어할 줄 수 설정

    def write(self, txt):
        super().write(txt)
        sys.__stdout__.write(txt)  # 터미널에도 출력
        lines = txt.splitlines()
        for line in lines:
            if line:
                self.output.append(line)
                # 로그 줄 수가 최대 길이를 초과하면 가장 오래된 로그부터 제거
                while len(self.output) > self.max_length:
                    self.output.pop(0)
                # 로그 줄 수가 자동 클리어 임계값을 초과하면 로그를 초기화
                if len(self.output) > self.auto_clear_threshold:
                    self.clear_output()

    def get_output(self):
        return '\n'.join(self.output)

    def clear_output(self):
        self.output = []

## logger instance
capture_stream = CaptureOutput()
sys.stdout = capture_stream

## Logger function
def get_captured_output():
    return capture_stream.get_output()


## yolo 실행
def run_yolo(input_video_url):
    global stop_execution
    stop_execution = False  # Reset the flag at the beginning of the function
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
        if stop_execution:
            print("Stopping YOLO processing...")
            break
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
    # resized_video = 'video/resized_/' + time.strftime('%Y%m%d%H%M') + '.mp4'
    # ffmpeg.input(output_path).output(resized_video, crf=35).run() # 영상 용량 축소
    return input_video_url, output_path


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
                btn_stop = gr.Button("Stop", size='sm')

            with gr.Column():
                output1 = gr.Video(autoplay=True) # 원본 비디오 재생

            with gr.Column():
                output2 = gr.Video(autoplay=True) # 결과 비디오 재생
             
            btn1.click(fn=run_yolo, inputs=input1, outputs=[output1, output2])
            btn_stop.click(fn=stop_functions)

        output_text = gr.Textbox(label="Progress Output", lines=20)
        demo.load(get_captured_output, None, output_text, every=1)

    demo.queue().launch(
            server_name=args.server_name,
            server_port=args.server_port,
            debug=True
        )