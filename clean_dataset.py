import os
TO_DELETE = ".DS_Store"

def clean_data(directory, depth=0):
	elements = os.listdir(directory)
	print(directory)
	if TO_DELETE in elements:
		print("removing ..  "+os.path.join(directory, elements[elements.index(TO_DELETE)]))
		os.remove(os.path.join(directory, elements[elements.index(TO_DELETE)]))
	if depth < 2:
		for el in elements:
			d = os.path.join(directory, el)
			if os.path.isdir(d):
				clean_data(d, depth+1)



def clean_data_old(directory):
    print(directory)
    elements = os.listdir(directory)
    if TO_DELETE in elements:
        print("removing ..  "+os.path.join(directory, elements[elements.index(TO_DELETE)]))
        os.remove(os.path.join(directory, elements[elements.index(TO_DELETE)]))
    for el in elements:
        d = os.path.join(directory, el)
        if os.path.isdir(d):
            clean_data(d)