KERNEL_NAME := low-rank-rnn-report
KERNEL_PREFIX := /tmp/low-rank-rnn-report-kernel
JUPYTER_PATH := $(KERNEL_PREFIX)/share/jupyter
NOTEBOOK_BUILD := /tmp/low-rank-rnn-report-source.ipynb

.PHONY: figures pdf

figures:
	mkdir -p /tmp/low-rank-rnn-ipython /tmp/low-rank-rnn-matplotlib
	env IPYTHONDIR=/tmp/low-rank-rnn-ipython ../.venv/bin/python -m ipykernel install \
		--prefix $(KERNEL_PREFIX) --name $(KERNEL_NAME) \
		--display-name "low-rank-rnn report build"
	../.venv/bin/python prepare_figures_notebook.py ../final.ipynb $(NOTEBOOK_BUILD)
	cd .. && env JUPYTER_PATH=$(JUPYTER_PATH) \
		IPYTHONDIR=/tmp/low-rank-rnn-ipython \
		MPLCONFIGDIR=/tmp/low-rank-rnn-matplotlib \
		jupyter nbconvert --execute --to notebook \
		--ExecutePreprocessor.kernel_name=$(KERNEL_NAME) \
		--ExecutePreprocessor.timeout=1800 \
		--output-dir=/tmp --output=low-rank-rnn-report.ipynb $(NOTEBOOK_BUILD)

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
