#!/bin/sh
# Builds the release archives into dist/release/: one zip (Windows) or
# tar.gz (Linux, macOS) per platform, each holding the single binary, a
# README (and the LICENSE if the repo has one), plus SHA256SUMS. CI runs
# this on a tag; run it by hand from the repo root to try it:
#   sh scripts/release.sh [version]
set -eu
name=assay
version=${1:-$(git describe --tags --always --dirty 2>/dev/null || echo dev)}
out=dist/release
rm -rf "$out" && mkdir -p "$out"
for target in linux/amd64 linux/arm64 windows/amd64 darwin/amd64 darwin/arm64; do
	os=${target%/*} arch=${target#*/}
	base=$name-$version-$os-$arch
	dir=$out/$base
	mkdir -p "$dir"
	exe=$name
	[ "$os" = windows ] && exe=$name.exe
	CGO_ENABLED=0 GOOS=$os GOARCH=$arch go build -trimpath \
		-ldflags "-s -w -X main.version=$version" -o "$dir/$exe" ./cmd/assay
	[ -f LICENSE ] && cp LICENSE "$dir/"
	if [ "$os" = windows ]; then
		# Notepad wants CRLF line endings.
		sed 's/$/\r/' tools/release/README-windows.txt > "$dir/README.txt"
		(cd "$out" && python3 -m zipfile -c "$base.zip" "$base")
	else
		cp tools/release/README-unix.txt "$dir/README.txt"
		tar -C "$out" -czf "$out/$base.tar.gz" "$base"
	fi
	rm -rf "$dir"
done
cd "$out"
if command -v sha256sum >/dev/null; then
	sha256sum -- *.zip *.tar.gz > SHA256SUMS
else
	shasum -a 256 -- *.zip *.tar.gz > SHA256SUMS
fi
ls -l
