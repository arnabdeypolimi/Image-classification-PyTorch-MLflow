FROM ubuntu:18.04
ENV LC_ALL=C
RUN apt-get update -y && apt-get install -y python3-pip python3-dev build-essential tesseract-ocr poppler-utils libsm6 libxext6 libxrender-dev libtesseract-dev libleptonica-dev pkg-config

# Install extras
#COPY requirements.yml /requirements.yml
COPY requirement.txt /requirement.txt
COPY . /code/
# If you are using a py27 image, change this to py27
#RUN /bin/bash -c ". activate py36 && conda env update -f=/requirements.yml"
CMD ["bash"]
RUN pip3 install -r /requirement.txt