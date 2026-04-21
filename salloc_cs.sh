#!/bin/bash
salloc --time 00:30:00 --nodes 1 --ntasks-per-node 1 --gpus 1 --mem 256g --qos cs --partition cs
