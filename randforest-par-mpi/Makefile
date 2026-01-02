CC = mpicc
#rufino@ipb.pt: removed debug and memory error detection options
#CFLAGS = -std=c99 -O1 -fsanitize=address -g -Wall -Wextra -Wpedantic

# Flags de warning
MUTE_CHECK_MAYBE_USED_UNINITIALIZED = -Wno-maybe-uninitialized
MUTE_CHECK_UNUSED_RESULT = -Wno-unused-result

# linha 1 para profiling, 2 para benchmarking
CFLAGS = -std=c99 -O0 -g -Wall -Wextra -Wpedantic $(MUTE_CHECK_MAYBE_USED_UNINITIALIZED) $(MUTE_CHECK_UNUSED_RESULT)
#CFLAGS = -std=c99 -O2 -Wall -Wextra -Wpedantic $(MUTE_CHECK_MAYBE_USED_UNINITIALIZED) $(MUTE_CHECK_UNUSED_RESULT)

MPIFLAGS = -I/share/apps/openmpi-4.1.4/include

MFLAGS = -lm

SRC = main.c \
      utils/utils.c \
      utils/data.c \
      utils/argparse.c \
      model/tree.c \
      model/forest.c \
      eval/eval.c \
      utils/log.c

OBJ = $(SRC:.c=.o)

TARGET = random-forest

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(MPIFLAGS) -o $@ $(OBJ) $(MFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(MPIFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
