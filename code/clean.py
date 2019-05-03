import os
count = 0
for file in os.listdir("../data/test_visit/test/"):
    if file.endswith(".npy"):
        os.remove("../data/test_visit/test/"+file)
        print("deleted", file)
