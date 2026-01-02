CC = gcc
#rufino@ipb.pt: removed debug and memory error detection options
#CFLAGS = -std=c99 -O1 -fsanitize=address -g -Wall -Wextra -Wpedantic

#rufino@ipb.pt: use 1st option for profiling, use 2nd option for benchmarking
CFLAGS = -std=c99 -O0 -g -Wall -Wextra -Wpedantic
#CFLAGS = -std=c99 -O2 -Wall -Wextra -Wpedantic


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
	$(CC) $(CFLAGS) -o $@ $(OBJ)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

test: $(TARGET)

	./$(TARGET) iris_binary.csv
