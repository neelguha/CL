
losses = []
for line in open("loss-logs/agent-0/losses.csv"):
    losses.append(float(line.split(",")[0]))


print sorted(losses)