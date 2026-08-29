# Build context must be data/datavalgen (the parent dir), not this folder,
# since this image reuses the same "example" model/factory package as
# ../Dockerfile. From data/datavalgen/:
#   docker build -f v6-validate/Dockerfile -t datavalgen-validate:local .
FROM ghcr.io/mdw-nl/datavalgen:v0.4.3

# Same model/factory package as ../Dockerfile, so get_model("example", ...)
# resolves here too.
ENV DATAVALGEN_DISTRIBUTION=datavalgen-test

COPY ./src /app/datavalgen-test/src
COPY ./pyproject.toml /app/datavalgen-test/pyproject.toml
RUN pip install --no-cache-dir --no-deps /app/datavalgen-test

# Federated orchestration layer on top of the datavalgen base image.
# vantage6-algorithm-tools pulls in psycopg2-binary, which has no prebuilt
# musl/Alpine wheel and must compile against libpq — hence the build deps,
# removed again once the wheel is built so they don't bloat the final image.
RUN apk add --no-cache --virtual .build-deps gcc musl-dev postgresql-dev \
    && pip install --no-cache-dir "vantage6-algorithm-tools==4.14.0" \
    && apk del .build-deps

ENV PKG_NAME=v6_validate
WORKDIR /app
COPY ./v6-validate/v6_validate.py .

# The base image sets ENTRYPOINT ["datavalgen"] for its own CLI; reset it so
# CMD below is what actually runs instead of being appended as an argument to
# "datavalgen".
ENTRYPOINT []
CMD ["python", "-c", "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"]
