FROM cs231a:project
MAINTAINER Luke Jaffe <lukejaffe1@gmail.com>

RUN pip3 install pytube
WORKDIR /test
RUN pytube http://youtube.com/watch?v=9bZkp7q19f0 --itag=22
RUN ls
