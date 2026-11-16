#!/bin/sh
# The one command that proves clip-analyzer works. Exit 0 = pass, anything else = fail.
# Written or approved by Nick. Agents may run it, never edit it.
# Checks: known 3D camera moves are labelled right, the upload off switch works, an uploaded clip is analysed.
cd "$(dirname "$0")" && exec python3 -I checks/check_all.py
