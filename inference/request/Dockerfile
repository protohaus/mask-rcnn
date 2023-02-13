FROM python:3.8
LABEL maintainer="tobias@protohaus.org"

ARG BUILD_DATE
ARG BUILD_VERSION

LABEL org.label-schema.build-date=$BUILD_DATE
LABEL org.label-schema.version=$BUILD_VERSION
LABEL org.label-schema.name="protohaus/ofai-web-inference"
LABEL org.label-schema.description="Inference of OFAI trained AI models"
LABEL org.label-schema.url="https://openfarming.ai/"

ARG USERNAME=deploy
ARG PROJECT_DIR=/home/$USERNAME/
RUN groupadd -r $USERNAME && useradd --no-log-init -r -g $USERNAME $USERNAME
RUN mkdir -p $PROJECT_DIR && chown $USERNAME:$USERNAME $PROJECT_DIR
WORKDIR $PROJECT_DIR
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

# Drop user privileges
USER $USERNAME

COPY Pipfile Pipfile.lock ./
RUN python3 -m pip install --user pipenv && pipenv install

COPY start.sh ./
COPY ./app ./app
COPY ./model ./model

EXPOSE 8000
STOPSIGNAL SIGINT
ENTRYPOINT [ "bash" ]
CMD [ "./start.sh" ]