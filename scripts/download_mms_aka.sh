#!/bin/bash
# Segmented parallel download of facebook/mms-tts-aka into a local model dir.
set -u
DEST="E:/UG-TTS-Task/05_models/mms-tts-aka"
mkdir -p "$DEST"
BASE="https://huggingface.co/facebook/mms-tts-aka/resolve/main"
UA="Mozilla/5.0 (research)"

# small files first
for f in config.json vocab.json special_tokens_map.json tokenizer_config.json README.md; do
  [ -s "$DEST/$f" ] || curl -sL --retry 5 --retry-delay 2 -A "$UA" -o "$DEST/$f" "$BASE/$f"
  echo "small: $f $(stat -c%s "$DEST/$f" 2>/dev/null)"
done

URL="$BASE/model.safetensors"
LEN=$(curl -sIL -A "$UA" "$URL" | grep -i '^content-length:' | tail -1 | tr -dc '0-9')
echo "model.safetensors total bytes: $LEN"
[ -z "$LEN" ] && { echo "no content-length; abort"; exit 1; }

NPART=8
PDIR="$DEST/parts"
mkdir -p "$PDIR"
CHUNK=$(( (LEN + NPART - 1) / NPART ))

dl_part () {
  local i=$1
  local start=$(( i * CHUNK ))
  local end=$(( start + CHUNK - 1 ))
  [ $end -ge $LEN ] && end=$(( LEN - 1 ))
  local out="$PDIR/part_$i"
  local want=$(( end - start + 1 ))
  for attempt in 1 2 3 4 5 6 7 8; do
    local have=0
    [ -f "$out" ] && have=$(stat -c%s "$out")
    [ "$have" -ge "$want" ] && return 0
    local from=$(( start + have ))
    curl -sL -A "$UA" --max-time 300 --speed-limit 20000 --speed-time 30 \
         -r "$from-$end" -o - "$URL" >> "$out" 2>/dev/null
  done
  have=$(stat -c%s "$out" 2>/dev/null || echo 0)
  [ "$have" -eq "$want" ] || { echo "PART $i INCOMPLETE $have/$want"; return 1; }
}

for i in $(seq 0 $((NPART-1))); do dl_part $i & done
wait

# verify + assemble
TOTAL=0
for i in $(seq 0 $((NPART-1))); do
  SZ=$(stat -c%s "$PDIR/part_$i" 2>/dev/null || echo 0)
  TOTAL=$((TOTAL + SZ))
done
echo "assembled size: $TOTAL / $LEN"
if [ "$TOTAL" -eq "$LEN" ]; then
  cat $(for i in $(seq 0 $((NPART-1))); do echo "$PDIR/part_$i"; done) > "$DEST/model.safetensors"
  rm -rf "$PDIR"
  echo "DONE: $(stat -c%s "$DEST/model.safetensors") bytes"
else
  echo "SIZE MISMATCH; keeping parts for resume"
  exit 1
fi