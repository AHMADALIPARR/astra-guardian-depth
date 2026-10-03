CC = cc
CFLAGS = -O2 -std=c11 -Wall -Wextra -Wpedantic -Werror
LDLIBS = -lm
SMALL = -DLT_D_MODEL=4 -DLT_N_HEADS=2 -DLT_D_FF=5 -DLT_MAX_T=8

.PHONY: test sanitize clean
test: test-small test-full
	./test-small
	./test-full

test-small: looped.c looped.h test_backend.c
	$(CC) $(CFLAGS) $(SMALL) looped.c test_backend.c $(LDLIBS) -o $@

test-full: looped.c looped.h test_backend.c
	$(CC) $(CFLAGS) looped.c test_backend.c $(LDLIBS) -o $@

train-looped: looped.c looped.h train_looped.c
	$(CC) $(CFLAGS) looped.c train_looped.c $(LDLIBS) -o $@

sanitize: looped.c looped.h test_backend.c
	$(CC) -O1 -g -std=c11 -Wall -Wextra -Wpedantic -Werror -fsanitize=address,undefined -fno-omit-frame-pointer looped.c test_backend.c $(LDLIBS) -o test-sanitize
	ASAN_OPTIONS=$${ASAN_OPTIONS:-detect_leaks=0} ./test-sanitize

clean:
	rm -f test-small test-full test-sanitize train-looped
