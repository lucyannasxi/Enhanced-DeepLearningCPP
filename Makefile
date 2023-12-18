include ./makes/defines.inc

.PHONY: release debug clean ctags format tidy help
.PHONY: library_release library_debug
.PHONY: tests_release tests_debug
.PHONY: samples_release samples_debug

SAMPLE_NAMES = toySample mnist mnist_conv cifar10_conv

all: release debug

release: library_release tests_release samples_release
debug: library_debug tests_debug samples_debug

#### library

library_release:
	@+make -C core release

library_debug:
	@+make -C core debug

#### tests

tests_release: library_release
	@+make -C tests release

tests_debug: library_debug
	@+make -C tests debug

#### samples

samples_release: librar