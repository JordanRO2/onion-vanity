#!/usr/bin/env bash
# Cross-check a generated <addr>.onion directory against a real tor (run under WSL / Linux).
#
# The check must confirm BOTH:
#   1. tor computes the SAME hostname from the key, and
#   2. tor leaves the key files byte-identical
# because tor silently generates a brand-new key (new hostname) if the tag bytes are corrupt — so a
# matching hostname alone is not enough.
#
# Network is disabled: tor still loads the hidden-service dir and derives/validates the hostname on
# startup, which is all this check needs. The live publish+fetch is a separate test.
#
# Usage:  bash scripts/tor-check.sh <path-to-addr.onion-dir>
set -euo pipefail

SRC="${1:?usage: tor-check.sh <addr.onion-dir>}"
command -v tor >/dev/null 2>&1 || { echo "FAIL: tor not installed (try: sudo apt install tor)"; exit 1; }
for f in hs_ed25519_secret_key hs_ed25519_public_key hostname; do
    [ -f "$SRC/$f" ] || { echo "FAIL: missing $SRC/$f"; exit 1; }
done

EXPECT="$(tr -d '\r\n' < "$SRC/hostname")"
WORK="$(mktemp -d)"
trap 'kill "${TOR_PID:-}" 2>/dev/null || true; wait 2>/dev/null || true; rm -rf "$WORK"' EXIT
HS="$WORK/hs"
DATA="$WORK/data"
mkdir -p "$HS" "$DATA"
cp "$SRC"/hs_ed25519_secret_key "$SRC"/hs_ed25519_public_key "$SRC"/hostname "$HS/"
chmod 700 "$HS" "$DATA"
chmod 600 "$HS"/hs_ed25519_secret_key "$HS"/hs_ed25519_public_key "$HS"/hostname

before="$(sha256sum "$HS/hs_ed25519_secret_key" "$HS/hs_ed25519_public_key" | awk '{print $1}')"

cat > "$WORK/torrc" <<EOF
SocksPort 0
DisableNetwork 1
DataDirectory $DATA
HiddenServiceDir $HS
HiddenServicePort 80 127.0.0.1:8080
Log notice file $WORK/tor.log
EOF

tor -f "$WORK/torrc" >/dev/null 2>&1 &
TOR_PID=$!
# Give tor a moment to load the hidden service and (re)write the hostname.
for _ in $(seq 1 20); do
    grep -q "Bootstrapped\|Opening\|rendezvous\|Configured hidden service\|Parsing GEOIP" "$WORK/tor.log" 2>/dev/null && break
    kill -0 "$TOR_PID" 2>/dev/null || break
    sleep 0.5
done
sleep 1
kill "$TOR_PID" 2>/dev/null || true
wait 2>/dev/null || true

got="$(tr -d '\r\n' < "$HS/hostname")"
after="$(sha256sum "$HS/hs_ed25519_secret_key" "$HS/hs_ed25519_public_key" | awk '{print $1}')"

rc=0
if [ "$got" != "$EXPECT" ]; then
    echo "FAIL: tor derived a different hostname"
    echo "  expected: $EXPECT"
    echo "  got:      $got"
    rc=1
fi
if [ "$before" != "$after" ]; then
    echo "FAIL: tor rewrote the key files (bad tags -> regenerated key)"
    rc=1
fi
if grep -qE '\[err\]' "$WORK/tor.log" 2>/dev/null; then
    echo "FAIL: tor logged errors:"
    grep -E '\[err\]' "$WORK/tor.log" | sed 's/^/  /'
    rc=1
fi
if grep -qE '\[warn\]' "$WORK/tor.log" 2>/dev/null; then
    echo "note: tor warnings (review; SocksPort 0 / DisableNetwork may warn benignly):"
    grep -E '\[warn\]' "$WORK/tor.log" | sed 's/^/  /'
fi

[ "$rc" -eq 0 ] && echo "OK: tor confirms $got"
exit "$rc"
