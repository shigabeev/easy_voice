def write_result(text, who):
    with open("data/history.txt", "a") as fs:
        fs.write(f'{who}: {text}')
