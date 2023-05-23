# syntax = edrevo/dockerfile-plus

INCLUDE+ ./Dockerfile.base

RUN apt-get update -y

RUN mkdir /dep/d4rl
COPY ./d4rl.zip /dep/d4rl/d4rl.zip
RUN cd /dep/d4rl && unzip /dep/d4rl/d4rl.zip && cd d4rl && pip install -e . --index https://mirrors.ustc.edu.cn/pypi/web/simple && cd /dep


RUN mkdir /dep/mjrl
COPY ./mjrl.zip /dep/mjrl/mjrl.zip
RUN cd /dep/mjrl && unzip /dep/mjrl/mjrl.zip && cd mjrl && pip install -e . --index https://mirrors.ustc.edu.cn/pypi/web/simple && cd /dep


COPY mujoco200.zip /dep/mujoco200.zip
RUN cd /dep && unzip mujoco200.zip && mkdir /root/.mujoco && mv mujoco200 /root/.mujoco/mujoco200
COPY ./mjkey.txt /root/.mujoco/mjkey.txt

COPY mujoco210.zip /dep/mujoco210.zip
RUN cd /dep && unzip mujoco210.zip && mv mujoco210 /root/.mujoco/mujoco210
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/root/.mujoco/mujoco200/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/root/.mujoco/mujoco210/bin"

COPY ./requirements.txt /dep/requirements.txt

RUN pip install -r /dep/requirements.txt --index https://mirrors.ustc.edu.cn/pypi/web/simple


RUN python -c "import d4rl"

CMD ["bash"]