id_to_char_array = ["a", "b"]

char_to_id_dict = dict()
for i, x in enumerate(id_to_char_array):
	char_to_id_dict[x] = i

def char_to_id(x):
	return char_to_id_dict[x]

def id_to_char(i):
	return id_to_char_array[i]
