import pandas as pd
import numpy as np
from math import floor
from timeit import default_timer as timer
from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHash, MinHashLSH
from sklearn.metrics.pairwise import cosine_similarity

pd.options.display.max_rows = 9999
df = pd.read_csv('Dataset.csv', encoding='unicode_escape')

names_awards_dblp_list = df[[df.columns[0], df.columns[1], df.columns[3]]].values.tolist()


class LSH:


    def __init__(self, num_hash_functions, num_buckets):
        self.num_hash_functions = num_hash_functions
        self.num_buckets = num_buckets
        self.hash_functions = self.create_hash_functions()

    def create_hash_functions(self):
        hash_functions = []
        for _ in range(self.num_hash_functions):
            a = np.random.randint(1, self.num_buckets)
            b = np.random.randint(0, self.num_buckets)
            hash_functions.append((a, b))
        return hash_functions

    def hash_value(self, value):
        hash_codes = []
        for a, b in self.hash_functions:
            hash_code = (a * hash(value) + b) % self.num_buckets
            hash_codes.append(hash_code)
        return hash_codes

    def index(self, data, tfidf_vectorizer):
        index = {}
        tfidf_matrix = tfidf_vectorizer.fit_transform(data)
        for i, item in enumerate(data):
            tfidf_hash = self.hash_value(tfidf_matrix[i])
            for j, h in enumerate(tfidf_hash):
                if h not in index:
                    index[h] = []
                index[h].append(i)
        return index

    def query(self, query_vector, tfidf_vectorizer, index):
        query_hash = self.hash_value(query_vector)
        candidate_set = set()
        for h in query_hash:
            if h in index:
                candidate_set.update(index[h])
        return candidate_set

        # Example usage:
        num_hash_functions = 50  # Number of hash functions
        num_buckets = 1000  # Number of buckets

        # Assuming you have a list of education texts
        education_data = filtered["Education"].tolist()

        # Create TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer()
        query_vector = tfidf_vectorizer.fit_transform(["your query education text"])

        # Create LSH index
        lsh_index = LSH(num_hash_functions, num_buckets)
        index = lsh_index.index(education_data, tfidf_vectorizer)

        # Query the LSH index
        candidate_set = lsh_index.query(query_vector, tfidf_vectorizer, index)


class MinimumBoundingObject:
    def __init__(self, low, high, child_values=None):


        self.low = low
        self.high = high
        self.child_values = child_values or []

class RTree:
    def __init__(self):
        self.names_awards_dblp_list = []  # Initialize an empty list to store the data points
        self.root = None

    def create_rtree(self, points, dimensions):
        M = 5
        upper_level_items = []

        points.sort(key=lambda point: point[0])

        while len(points) > M:
            group_length = min(M, len(points))
            new_minimum_bounding_object = self.minimum_bounding_object_calculator(points[:group_length], dimensions)
            new_minimum_bounding_object.child_values = points[:group_length]
            upper_level_items.append(new_minimum_bounding_object)
            points = points[group_length:]

        if points:
            new_minimum_bounding_object = self.minimum_bounding_object_calculator(points, dimensions)
            new_minimum_bounding_object.child_values = points
            upper_level_items.append(new_minimum_bounding_object)

        if len(upper_level_items) <= M:
            return upper_level_items  # This line should return a list of items
        else:
            return self.create_rtree(upper_level_items, dimensions)

    def minimum_bounding_object_calculator(self, points, dimensions):
        lower = ['zzzzzzz'] * dimensions
        upper = [''] * dimensions
        child_values = []

        for point in points:
            for d in range(dimensions):
                coord = str(point[d])  # Convert coordinate to string
                lower[d] = min(lower[d], coord)
                upper[d] = max(upper[d], coord)

            child_values.append(point)

        return MinimumBoundingObject(lower, upper, child_values)

    def find_scientist(self, letter_range, min_awards, min_dblp_record, max_dblp_record):
        target_node = [letter_range, min_awards, min_dblp_record, max_dblp_record]
        result_list = []
        self.query(self.root, target_node, result_list)

        if result_list:
            print("Scientist found in the R-tree.")
            return result_list
        else:
            print("Scientist not found in the R-tree.")
            return None

    def query(self, node, target_node, result_list):
        if node is None:
            return

        print(f"Checking node: {node}, Target: {target_node}")

        #print(f"Checking node: {node}")

        if isinstance(node, MinimumBoundingObject):
            for child in node.child_values:
                self.query(child, target_node, result_list)
        elif isinstance(node, list):
            for child in node:
                print(f"Comparing with child: {child}")
                if child == target_node:
                    result_list.append(child)


    def insert_recursive(self, node, point):
        if node is None:
            return self.minimum_bounding_object_calculator([point], dimensions=3)

        if isinstance(node, MinimumBoundingObject):
            for item in node.child_values:
                if point[2] >= int(item.low[2]) and point[2] <= int(item.high[2]):
                    new_children = self.insert_recursive(item, point)
                    if new_children:
                        item.child_values.append(new_children)
                        self.update_node_bounds(item)  # Update bounds of the parent node
            if len(node.child_values) < 4:
                node.child_values.append(point)
                self.update_node_bounds(node)  # Update bounds of the parent node
            else:
                new_min_bounding_objs = []
                for item in node.child_values:
                    new_min_bounding_obj = self.minimum_bounding_object_calculator([item], dimensions=3)
                    new_min_bounding_obj.child_values.append(item)
                    new_min_bounding_objs.append(new_min_bounding_obj)
                new_min_bounding_obj = self.minimum_bounding_object_calculator([point], dimensions=3)
                new_min_bounding_obj.child_values.append(point)
                new_min_bounding_objs.append(new_min_bounding_obj)
                return new_min_bounding_objs
        else:
            if len(node) < 4:
                node.append(point)
            else:
                new_min_bounding_objs = []
                for item in node:
                    new_min_bounding_obj = self.minimum_bounding_object_calculator([item], dimensions=3)
                    new_min_bounding_obj.child_values.append(item)
                    new_min_bounding_objs.append(new_min_bounding_obj)
                new_min_bounding_obj = self.minimum_bounding_object_calculator([point], dimensions=3)
                new_min_bounding_obj.child_values.append(point)
                new_min_bounding_objs.append(new_min_bounding_obj)
                return new_min_bounding_objs

        return node

    def delete_recursive(self, node, x_coord, y_coord, z_coord):
        if isinstance(node, MinimumBoundingObject):
            new_children = []
            delete_node = False

            for item in node.child_values:
                if (
                        z_coord >= int(item.low[2]) and
                        z_coord <= int(item.high[2])
                ):
                    delete_node = True
                    new_children.extend(self.delete_recursive(item, x_coord, y_coord, z_coord))
                else:
                    new_children.append(item)

            node.child_values = new_children
            if not node.child_values:
                return [], delete_node

            for d in range(3):
                node.low[d] = 'zzzzzzz'
                node.high[d] = ''
                for item in node.child_values:
                    node.low[d] = min(node.low[d], item.low[d])
                    node.high[d] = max(node.high[d], item.high[d])

            return [node], delete_node

        else:
            new_children = []
            delete_node = False

            for item in node:
                if item[0] == x_coord and item[1] == y_coord and item[2] == z_coord:
                    delete_node = True
                else:
                    new_children.append(item)

            return new_children, delete_node


    def delete(self, x_coord, y_coord, z_coord):
        self.names_awards_dblp_list, _ = self.delete_recursive(self.names_awards_dblp_list, x_coord, y_coord, z_coord)
        self.root = self.create_rtree(self.names_awards_dblp_list, dimensions=3)


def split_node(self, points, dimensions):
    M = 5
    upper_level_items = []

    points.sort(key=lambda point: point[2])

    while len(points) > M:
        group_length = min(M, len(points))
        new_minimum_bounding_object = self.minimum_bounding_object_calculator(points[:group_length], dimensions)
        new_minimum_bounding_object.child_values = points[:group_length]
        upper_level_items.append(new_minimum_bounding_object)
        points = points[group_length:]

    if points:
        new_minimum_bounding_object = self.minimum_bounding_object_calculator(points, dimensions)
        new_minimum_bounding_object.child_values = points
        upper_level_items.append(new_minimum_bounding_object)

    return upper_level_items

def update_node_bounds(self, node):
    for d in range(3):
        node.low[d] = min(item.low[d] for item in node.child_values)
        node.high[d] = max(item.high[d] for item in node.child_values)


    def delete_recursive(self, node, x_coord, y_coord, z_coord):
        if isinstance(node, MinimumBoundingObject):
            new_children = []
            for item in node.child_values:
                if (
                        z_coord >= int(item.low[2]) and
                        z_coord <= int(item.high[2])
                ):
                    new_children = self.delete_recursive(item, x_coord, y_coord, z_coord)
                else:
                    new_children.append(item)

            node.child_values = new_children
            if not node.child_values:
                return None

            for d in range(3):
                node.low[d] = 'zzzzzzz'
                node.high[d] = ''
                for item in node.child_values:
                    node.low[d] = min(node.low[d], item[d])
                    node.high[d] = max(node.high[d], item[d])

            return node

        else:
            new_children = []
            for item in node:
                if item[0] != x_coord or item[1] != y_coord or item[2] != z_coord:
                    new_children.append(item)

            if not new_children:
                return None

            return new_children

    def delete(self, x_coord, y_coord, z_coord):
        self.names_awards_list = self.delete_recursive(self.names_awards_list, x_coord, y_coord, z_coord)
        self.root = self.create_rtree(self.names_awards_list, dimensions=3)


def print_tree(node, level=0, prefix="Root: "):
    if isinstance(node, MinimumBoundingObject):
        print(" " * (level * 4) + prefix + f"Low: {node.low}, High: {node.high}")
        for child in node.child_values:
            if isinstance(child, MinimumBoundingObject):
                print_tree(child, level + 1, prefix="Child: ")
            else:
                print(" " * ((level + 1) * 4) + f"Leaf: {child}")
    elif isinstance(node, list):
        for child in node:
            if isinstance(child, MinimumBoundingObject):
                print_tree(child, level, prefix="List Child: ")
            elif isinstance(child, list):
                print_tree(child, level, prefix="List Child: ")
            else:
                print(" " * (level * 4) + prefix + f"Leaf: {child}")
    else:
        print("Unexpected node type:", type(node))


def update_rtree_and_list_3d(rtree_instance, names_awards_dblp_list):
    rtree_instance.names_awards_dblp_list = names_awards_dblp_list
    rtree_instance.root = rtree_instance.create_rtree(names_awards_dblp_list, dimensions=3)





num_hash_functions = 50  # Number of hash functions
num_buckets = 1000  # Number of buckets
root = None
choice=-1
rtree_instance = RTree()
#root = rtree_instance.create_rtree(names_awards_dblp_list, dimensions=3)


while choice!=0:
    print("Give a number for each choice below: ")
    print("0) End the program")
    print("1) Create the R tree")
    print("2) Print the R tree")
    print("3) Insert a node")
    print("4) Delete a node")
    print("5) Search for a node")
    print("6) Update a node")
    print("7) Query search")
    choice=int(input())

    if choice==1:# CREATE THE TREE
        start_time = timer()
        root = rtree_instance.create_rtree(names_awards_dblp_list, dimensions=3)
        print("Root after tree creation:", root)
        end_time = timer()
        execution_time = end_time - start_time
        print("\nCreate R Tree executed in: ", execution_time, " seconds")
        print_tree(root)

    elif choice == 2:  # PRINT THE TREE
        start_time = timer()
        update_rtree_and_list_3d(rtree_instance, names_awards_dblp_list)
        end_time = timer()
        execution_time = end_time - start_time
        print("\nCreate R Tree executed in: ", execution_time, " seconds")
        print_tree(rtree_instance.root)

    elif choice == 3:  # INSERT A NODE
        print("Give the coordinates to insert:")
        insert_name = input("Enter Name: ")
        insert_awards = int(input("Enter Number of Awards: "))
        insert_dblp = int(input("Enter DBLP Record: "))

        start_time = timer()
        rtree_instance.root = rtree_instance.insert_recursive(rtree_instance.root, [insert_name, insert_awards, insert_dblp])
        rtree_instance.names_awards_dblp_list.append([insert_name, insert_awards, insert_dblp])  # Append the new point
        update_rtree_and_list_3d(rtree_instance, rtree_instance.names_awards_dblp_list)
        end_time = timer()

        execution_time = end_time - start_time
        print("The updated R-tree after insertion: ")
        print_tree(rtree_instance.root)
        print("\nInsert node executed in: ", execution_time, " seconds")

    elif choice == 4:  # DELETE A NODE
        print("Give the coordinates to delete:")
        delete_name = input("Enter Name: ")
        delete_awards = int(input("Enter Number of Awards: "))
        delete_dblp = int(input("Enter DBLP Record: "))

        start_time = timer()
        rtree_instance.delete(delete_name, delete_awards, delete_dblp)
        update_rtree_and_list_3d(rtree_instance, rtree_instance.names_awards_dblp_list)
        end_time = timer()

        execution_time = end_time - start_time
        print("The updated R-tree after deletion: ")
        print_tree(rtree_instance.root)
        print("\nDelete node executed in: ", execution_time, " seconds")

    # Inside the main loop
    elif choice == 5:  # SEARCH FOR A SCIENTIST
        name = input("Enter scientist's name: ")
        awards = int(input("Enter number of awards: "))
        dblp_record = int(input("Enter scientist's DBLP record: "))

        result = rtree_instance.find_scientist(name, awards, dblp_record)
        if result:
            print("Scientist found in the R-tree:", result)
        else:
            print("Scientist not found in the R-tree.")

    elif choice == 6:  # UPDATE A NODE
        print("Enter the details of the scientist to update:")
        old_name = input("Enter scientist's current name: ")
        old_awards = int(input("Enter scientist's current number of awards: "))
        old_dblp = int(input("Enter scientist's current DBLP record: "))

        new_name = input("Enter scientist's new name: ")
        new_awards = int(input("Enter scientist's new number of awards: "))
        new_dblp = int(input("Enter scientist's new DBLP record: "))

        start_time = timer()

        # First, delete the old node
        rtree_instance.delete(old_name, old_awards, old_dblp)

        # Then insert the new node with updated information
        rtree_instance.root = rtree_instance.insert_recursive(rtree_instance.root, [new_name, new_awards, new_dblp])

        # Update the list and R-tree
        rtree_instance.names_awards_dblp_list.append([new_name, new_awards, new_dblp])
        update_rtree_and_list_3d(rtree_instance, rtree_instance.names_awards_dblp_list)

        end_time = timer()

        execution_time = end_time - start_time
        print("The updated R-tree after updating the node: ")
        print_tree(rtree_instance.root)
        print("\nUpdate node executed in: ", execution_time, " seconds")


    elif choice == 7:
        letter_range = input("Enter the letter range (e.g. [A,G]): ")
        letter_range = list(letter_range.strip("[]").split(","))
        letter_range = [letter_range[0], letter_range[-1]]
        min_awards = int(input("Enter the minimum number of awards "))
        min_dblp_record = input("Enter the minimum DBLP record ")
        max_dblp_record = input("Enter the maximum DBLP record ")

        bounds = [(0, 0), (98, 98)]
        tree = RTree()

        dataset = pd.read_csv('Dataset2.csv', encoding='unicode_escape')
        # Define and filter the dataset
        filtered = dataset[
            (dataset["Surname"].apply(lambda x: letter_range[0] <= x[0] <= letter_range[1])) &
            (dataset["#Awards"] > min_awards) &
            (min_dblp_record <= dataset["#DBLP_Record"]) & (dataset["#DBLP_Record"] <= max_dblp_record)
            ]

        # Extract education data from filtered DataFrame
        education_data = filtered["Education"].tolist()

        start_time = timer()
        results = tree.find_scientist(letter_range, min_awards, min_dblp_record, max_dblp_record)
        end_time = timer()

        print("Scientists with surnames in the range {range}, more than {awards} awards, and DBLP record between {min_dblp} and {max_dblp}:".format(
            range=letter_range, awards=min_awards, min_dblp=min_dblp_record, max_dblp=max_dblp_record))
        for scientist in results:
            print(scientist["Surname"], scientist["#Awards"], scientist["#DBLP_Record"])

        num_hash_functions = 50  # Number of hash functions
        num_buckets = 1000  # Number of buckets

        similarity_percentage = float(input("Enter the similarity percentage (0.0-100.0): ")) / 100.0

        #  LSH
        lsh_index = LSH(num_hash_functions, num_buckets)


        tfidf_vectorizer = TfidfVectorizer()
        query_vector = tfidf_vectorizer.fit_transform(["your query education text"])

        # Create LSH index
        index = lsh_index.index(education_data, tfidf_vectorizer)

        # Query
        candidate_set = lsh_index.query(query_vector, tfidf_vectorizer, index)

        similar_education_results = []
        for i in candidate_set:
            scientist = df.loc[i]
            similar_education_results.append((scientist["Surname"], scientist["Education"]))
            print(scientist["Surname"], scientist["Education"])
            print(i)

        execution_time = end_time - start_time
        print("\nQuery search executed in: ", execution_time, " seconds")




    else:
        con = False