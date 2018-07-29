import csv
from pathlib import Path
import os.path
results = []
pair=[]
f = open('text.txt', 'w')
results.append(["filepath","label"])
with open('lfw_allnames.csv') as File:
    reader = csv.DictReader(File)
    for row in reader:
        new_row=dict(row)
        print(row)
        for i in range(int(new_row["images"])):
            n=i+1
            number=str(n)
            print(type(number))
            print(type(new_row["name"]))
            if (int(new_row["images"])>9):
                name=new_row["name"]+"_"+"00"+number+".jpg"
            else:
                name = new_row["name"] + "_" + "000" + number + ".jpg"
            str_path=os.path.join("D:\\EXEFiles\\lfw-deepfunneled\\lfw-deepfunneled2\\lfw-deepfunneled\\", new_row["name"],name)
            f.write(str_path + '\n')
            pair=[str_path,new_row["name"]]
            results.append(pair)
            pair=[]
    print(results)



myFile = open('example3.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(results)

print("Writing complete")