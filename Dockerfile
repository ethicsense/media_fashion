FROM python

LABEL app_name="fashioncaster"
LABEL version="0.2"
LABEL command="docker run -d -it --name fashioncaster -p 8779:8779 evcrepo/fashioncaster:x86_64-0.2"

WORKDIR /home
ADD ./model.tar.gz .

RUN mkdir -p video
RUN mkdir -p video/out
RUN touch output.log

RUN apt-get update && apt-get install -y sudo
RUN chmod +w /etc/sudoers
RUN echo 'irteam ALL=(ALL) NOPASSWD:ALL' | tee -a /etc/sudoers
RUN chmod -w /etc/sudoers
RUN sudo apt-get install -y libgl1-mesa-glx
RUN sudo apt-get install -y python3-pip
RUN sudo apt-get install -y ffmpeg

# Install Packages
RUN pip install ffmpeg-python
RUN pip install --upgrade pip

RUN pip install ultralytics==8.2.49
RUN pip install ultralytics-thop==2.0.0

RUN pip install torch==2.3.1
RUN pip install torchaudio==2.3.1
RUN pip install torchvision==0.18.1

RUN pip install gradio

ENTRYPOINT ["python3", "app.py"]
CMD ["--server_port", "8779"]