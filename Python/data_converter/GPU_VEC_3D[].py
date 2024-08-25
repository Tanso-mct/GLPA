def parse_line(line):
    xIndex = line.find("x=") + 2
    yIndex = line.find("y=") + 2
    zIndex = line.find("z=") + 2

    x = float(line[xIndex:line.find(" ", xIndex)])
    y = float(line[yIndex:line.find(" ", yIndex)])
    z = float(line[zIndex:line.find(" ", zIndex)])

    return {"x": x, "y": y, "z": z}
    

data = []
with open("data_converter/input.txt", "r") as f:
    for line in f:
        parsed = parse_line(line)
        data.append(parsed)


with open("data_converter/output.txt", "w") as f:
    for line in data:
        f.write(str(line["x"]) + "," + str(line["y"]) + "," + str(line["z"]) + "\n")
