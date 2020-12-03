import json

file = open('D:\\Desktop\\dataset\\FLIR_ADAS_1_3 1\\FLIR_ADAS_1_3\\train\\thermal_annotations.json')
data = json.load(file) 

print(data.keys())
print(len(data['annotations']))
for i in range(12):
	print(data['annotations'][i])

#print(data)