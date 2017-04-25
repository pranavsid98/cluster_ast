import betterast, ast, zss, codegen, astdump, types, json, numpy
import sklearn.cluster

class WeirdNode(betterast.Node):
	def get_children(node):
		return node.children

	def get_label(node):
		return node.label

def dist(tree1,tree2):
	return zss.simple_distance(tree1, tree2, WeirdNode.get_children, WeirdNode.get_label)

def make_tree(tree):
	A = WeirdNode(str(tree))
	for node in ast.walk(tree):
		A = A.addkid(WeirdNode(str(node)))
	return A

def process(temp_str):
	temp = ""
	for line in (temp_str.replace("\\n","\n").replace("\\t","\t")).splitlines():
		if not line.startswith("#"):
			temp += line
			temp += "\n"
	return temp	

f = open("euler-data.txt")
f_str = f.read()
f_str = f_str.split('\"')
i=0
list_code=[]
for line in f_str:
	if i != 0 and i <=150:
		if line.startswith(": 1"):
			continue
		list_code.append(line)
	i+=1

#data = {}
#with open('euler_refined.json') as fp:
#    data = json.load(fp)

#list_code = []
#for key in data:
#	list_code.append(key)


for i in range(len(list_code)):
	list_code[i] = process(list_code[i])

list_code2 = []
for i in range(len(list_code)):
	try:
		list_code2.append(make_tree(ast.parse(list_code[i])))
	except:
		continue

list_code2 = numpy.asarray(list_code2)
edit_similarity = -1*numpy.array([[dist(tree1, tree2) for tree1 in list_code2] for tree2 in list_code2])

affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
affprop.fit(edit_similarity)
for cluster_id in numpy.unique(affprop.labels_):
	exemplar = list_code[affprop.cluster_centers_indices_[cluster_id]]
	temp = []
	for i in range(len((numpy.nonzero(affprop.labels_==cluster_id))[0])):
		temp.append(list_code[i])
	cluster = numpy.unique(list_code2[numpy.nonzero(affprop.labels_==cluster_id)])
	cluster_str = ", ".join(temp)
	number_ele = len(temp)
	print (" - *%s:* %d" % (exemplar, number_ele))


