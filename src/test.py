from agents.db import vector

# print(vector.get_contexts("class_data"))
# vector.add_vector()

res = vector.search_vector("vector-class-data", "python")
print(res)

