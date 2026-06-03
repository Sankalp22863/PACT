#!/bin/bash
export GID=$(id -g)

# Xyce 7.10.0 (serial, RHEL8) - extracted to ~/xyce/
export XYCE_HOME=/home/sgnaik/xyce/usr/local/XyceNF_7.10
export PATH=$XYCE_HOME/bin:$PATH
