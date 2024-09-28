
make run ARGS="-d \"../data/\" -ds \"./output/weights/\" -dl \"./output/weights/weights_200.bin\" -w 28 -h 28 -fh 3 -fw 3 -sh 1 -sw 1 -ph 1 -pw 1 -nf 64 -hd 64 -nc 10 -bs 128 -lr 0.001 -e 100" > output_log.txt 2>&1
