FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake openmpi-bin libopenmpi-dev python3 python3-matplotlib \
 && rm -rf /var/lib/apt/lists/*

# Docker's default seccomp profile blocks the cross-memory-attach syscalls
# used by Open MPI's shared-memory transport.
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none

# Open MPI does not run as root by default; use the image's unprivileged user.
USER ubuntu
WORKDIR /home/ubuntu/mpi-allreduce
COPY --chown=ubuntu:ubuntu . .

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j

CMD ["ctest", "--test-dir", "build", "--output-on-failure"]
