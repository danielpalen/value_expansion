FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

MAINTAINER Daniel Palenicek

# Install Conda
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda --version && /bin/bash -c "conda init bash"

RUN mkdir value_expansion
WORKDIR /value_expansion

RUN conda create -n value_expansion python=3.7

# Make RUN command user the value_expansion conda env.
SHELL ["conda", "run", "-n", "value_expansion", "/bin/bash", "-c"]
RUN echo 'eval $(conda shell.bash hook) && conda activate value_expansion' >> ~/.bashrc

RUN pip install --upgrade pip 

# Brax Dependencies
COPY brax brax
RUN cd brax && pip install -e .[develop]

# Value Expansion Dependencies
COPY setup.py setup.py
RUN mkdir model_based_rl && pip install -e . && rm -r model_based_rl

COPY . .
WORKDIR /value_expansion/scripts/experiments
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "value_expansion"]

CMD python run_experiment.py --help