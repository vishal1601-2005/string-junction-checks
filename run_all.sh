#!/usr/bin/env bash
# Reproduces every number quoted in the manuscript. Outputs go to results/.
set -e
mkdir -p results
for s in modes conv zeta extra_checks lz_check lz_iso series_and_quartic trivalent closed_channel closed_channel_general exact_closed unequal_arm_M casimir_force displacement_sector fusion; do
  echo "== running $s.py"; python3 $s.py > results/$s.txt 2>&1
done
echo "done; see results/*.txt"
