#!/usr/bin/env bash
# Run one training arm, keeping the FULL output and showing only the summary.
#
# The pattern every round_*.sh used was:
#   python train.py ... 2>&1 | grep -E "^\[|entsec-gold|trained in" || echo "  FAILED"
#
# which loses failures twice over. stderr is merged into the pipe and then
# discarded by a grep that does not match tracebacks, and the `|| echo FAILED`
# never fires because grep DOES match the "[transformers] LOAD REPORT" banner on
# `^\[`. Round Z lost three arms this way: they exited leaving an empty
# checkpoint directory and the log showed only their headers.
#
# Usage: runarm.sh <tag> <command...>
set -u
tag="$1"; shift
log="/tmp/arm-${tag}.log"
"$@" > "$log" 2>&1
rc=$?
grep -E "^\[${tag}|entsec-gold|real-gold|refined-gold|trained in" "$log"
if [ $rc -ne 0 ]; then
  echo "  FAILED rc=$rc — last lines of $log:"
  grep -viE "it/s\]|^\s*$|Map:" "$log" | tail -12 | sed 's/^/    /'
fi
