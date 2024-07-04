# Hackathon!

Use **make** to build the program

Use `for i in $(seq 0 10019); do cp "$(printf "./tensors/%02iout.txt" "$((i % 52 + 1))")" "$(printf "./tensors_t/%05iout.txt" "$((i + 1))")"; done` to create a sub-sample and test on it