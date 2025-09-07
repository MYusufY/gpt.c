CC = gcc

run: run.c
	$(CC) -O3 -o run run.c -lm

debug: run.c
	$(CC) -g -o run run.c -lm

clean:
	rm -f run