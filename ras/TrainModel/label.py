# class Hierarcy_Builder():
#     def __init__(self):
#         hierarcy_dict = {}
#         tree_height = 0
    
#     def set_children(self, key):
#         pass

#     def recursive_depth_search(self, root_dict, key):
#         end = False
#         path = None
#         if type(root_dict) is list:
#             end = True
#             children = root_dict
#         else:
#             children = list(root_dict.keys())

#         for child in children:
#             found = None
#             if not end:
#                 path = self.recursive_depth_search(root_dict[child], key)
#                 if path:
#                     path.append(child)
#                     return path
                
#             if key in child or found:
#                 if path is None:
#                     path = [child]
#                 else:
#                     path.append(child)
#                 return path


#         if child != key and end == True:
#             return



#     def get_hierarcy(self, label_list):
#         print(label_list)
#         if len(label_list) == 1:
#             return label_list[0]

#         hierarcy = self.get_hierarcy(label_list[:-1])

#         return {label_list[-1]:hierarcy}
#         # return {hierarcy:{}}

        
#         # return {label_list[-1]:{hierarcy}}
        
# # {"animals":{"reptiles":{"turtles"}}}


# class Hierarchical_Label:
#   def __init__(self, data, label_dict):
#     pass


# if __name__ == "__main__":
#     test = Hierarcy_Builder()

#     test_dict = {"animal"}

#     test.hierarcy_dict = {
#         "animals": {
#             "mammals": {
#                 "carnivores": ["cat", "dog", "wolf"],
#                 "herbivores": ["mouse", "rabbit", "deer"]
#             },
#             "birds": {
#                 "perching birds": ["parakeet", "dove", "pigeon"],
#                 "waterbirds": ["duck", "swan", {"goose":"head"}]
#             },
#             "reptiles": ["lizard", "snake", "turtle"]
#         },
#         "plants": {
#             "trees": ["maple", "oak", "pine"],
#             "shrubs": ["azalea", "rhododendron", "juniper"],
#             "flowers": ["daisy", "rose", "violet"]
#         },
#         "minerals": {
#             "metals": ["gold", "silver", "copper"],
#             "non-metals": ["carbon", "oxygen", "nitrogen"],
#             "minerals": ["quartz", "calcite", "gypsum"]
#         }
#     }

#     recursive_path = test.recursive_depth_search(test.hierarcy_dict, "head")
#     print(test.get_hierarcy(recursive_path))