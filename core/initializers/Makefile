include ../../makes/defines.inc

OUTDIR_R=$(OUTDIR)/lib_objs_r
OUTDIR_D=$(OUTDIR)/lib_objs_d

CPP_SOURCES=$(wildcard *.cpp)
CPP_HEADERS=$(wildcard *.h)
OBJS_R=$(patsubst %.cpp,$(OUTDIR_R)/%.o,$(CPP_SOURCES))
OBJS_D=$(patsubst %.cpp,$(OUTDIR_D)/%.o,$(CPP_SOURCES))

TIDY_TARGETS=$(patsubst %.cpp, .tidy_%,$(CPP_SOURCES))

COMMON_CFLAGS += -Wno-unused-parameter -I$(ROOT_DIR)/core -fPIC
CFLAGS_R=$(COMMON_CFLAGS) -O3
CFLAGS_D=$(COMMON_CFLAGS) -g

.PHONY: release debug format

all: release debug

release: $(OBJ