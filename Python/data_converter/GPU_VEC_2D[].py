OUTPUT_XZ = False
OUTPUT_YZ = True

def parse_line(line):
    xIndex = line.find("x=") + 2
    yIndex = line.find("y=") + 2

    x = float(line[xIndex:line.find(" ", xIndex)])
    y = float(line[yIndex:line.find(" ", yIndex)])

    return {"x": x, "y": y}
    

data = []
with open("data_converter/input.txt", "r") as f:
    for line in f:
        parsed = parse_line(line)
        data.append(parsed)


with open("data_converter/output.txt", "w") as f:
    for line in data:
        if OUTPUT_XZ:
            f.write(str(line["x"]) + ",0.0," + str(line["y"]) + "\n")
        elif OUTPUT_YZ:
            f.write("0.0," + str(line["x"]) + "," + str(line["y"]) + "\n")
