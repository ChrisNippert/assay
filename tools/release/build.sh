#!/bin/sh
# Builds ollamui for other machines into dist/: one zip (Windows) or
# tar.gz (Linux, macOS) per platform, each holding the single binary and
# a README. Run from the repo root: sh tools/release/build.sh [version]
set -eu
version=${1:-$(git describe --tags --always --dirty 2>/dev/null || echo dev)}
rm -rf dist && mkdir -p dist
for target in windows/amd64 linux/amd64 linux/arm64 darwin/arm64; do
	os=${target%/*} arch=${target#*/}
	name=ollamui-$version-$os-$arch
	dir=dist/$name
	mkdir -p "$dir"
	exe=ollamui
	[ "$os" = windows ] && exe=ollamui.exe
	CGO_ENABLED=0 GOOS=$os GOARCH=$arch go build -trimpath \
		-ldflags "-s -w -X main.version=$version" -o "$dir/$exe" ./cmd/ollamui
	if [ "$os" = windows ]; then
		# Notepad wants CRLF line endings.
		sed 's/$/\r/' tools/release/README-windows.txt > "$dir/README.txt"
		(cd dist && python3 -m zipfile -c "$name.zip" "$name")
	else
		cp tools/release/README-unix.txt "$dir/README.txt"
		tar -C dist -czf "dist/$name.tar.gz" "$name"
	fi
	rm -rf "$dir"
done
ls -l dist
