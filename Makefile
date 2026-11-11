PKGS    = libpipewire-0.3 libavformat libavcodec libavutil
CFLAGS  ?= -O2 -g -Wall -Wextra -Wno-unused-parameter
CFLAGS  += $(shell pkg-config --cflags $(PKGS))
LDLIBS  += $(shell pkg-config --libs $(PKGS)) -lpthread -lm
PREFIX  ?= $(HOME)/.local

pwrec: pwrec.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

install: pwrec
	install -Dm755 pwrec $(DESTDIR)$(PREFIX)/bin/pwrec

clean:
	rm -f pwrec

.PHONY: install clean
