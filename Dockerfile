FROM debian:buster-slim AS intermediate
# using buster because that's what worked for me
# Install dependencies
RUN apt-get -qq update; \
    apt-get install -qqy --no-install-recommends \
        gnupg2 wget ca-certificates apt-transport-https \
        autoconf automake cmake dpkg-dev file make patch libc6-dev

# Install LLVM
RUN echo "deb https://apt.llvm.org/buster llvm-toolchain-buster-16 main" \
        > /etc/apt/sources.list.d/llvm.list && \
    wget -qO /etc/apt/trusted.gpg.d/llvm.asc \
        https://apt.llvm.org/llvm-snapshot.gpg.key && \
    apt-get -qq update && \
    apt-get install -qqy -t llvm-toolchain-buster-16 clang-16 clang-tidy-16 clang-format-16 lld-16 lldb-16 libc++-16-dev libc++abi-16-dev && \
    for f in /usr/lib/llvm-16/bin/*; do ln -sf "$f" /usr/bin; done && \
    rm -rf /var/lib/apt/lists/*

FROM intermediate as test

COPY tests /tests

RUN /tests/run.sh 16

FROM intermediate as final