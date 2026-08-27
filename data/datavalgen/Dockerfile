FROM ghcr.io/mdw-nl/datavalgen:v0.4.3

# default model and factory
ENV DATAVALGEN_DISTRIBUTION=datavalgen-test
ENV DATAVALGEN_MODEL=example
ENV DATAVALGEN_FACTORY=example

COPY ./src /app/datavalgen-test/src
COPY ./pyproject.toml /app/datavalgen-test/pyproject.toml

# install our example data model package; datavalgen (already installed in
# the base image) finds it via python entry-points
RUN pip install --no-cache-dir --no-deps /app/datavalgen-test
