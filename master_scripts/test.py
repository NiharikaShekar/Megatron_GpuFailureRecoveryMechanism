log_path = "train_run.log"

with open(log_path, "r") as f:
    for line_num, line in enumerate(f, 1):
        if "training ..." in line:
            print(f"Found at line {line_num}: {line.strip()}")
	else:
	    print("Not training")
