import pandas
import numpy as np
from timeit import default_timer as timer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# pairnei dedomena apo to dataset
dataset = pandas.read_csv('BigDataset.csv', encoding='unicode_escape')


def filter_scientists(letter_range, min_awards, min_dblp_record, max_dblp_record):
    filtered = dataset[
        (dataset["Surname"].apply(lambda x: x[0] >= letter_range[0] and x[0] <= letter_range[-1])) &
        (dataset["#Awards"] > min_awards) &
        (dataset["#DBLP_Record"] >= min_dblp_record) & (dataset["#DBLP_Record"] <= max_dblp_record)
        ]
    return filtered

# Function gia dimiourgia LSH
def create_hash_functions(num_functions, num_buckets):
    hash_functions = []
    for _ in range(num_functions):
        a = np.random.randint(1, num_buckets)
        b = np.random.randint(0, num_buckets)
        hash_functions.append((a, b))
    return hash_functions

def hash_value(value, a, b, num_buckets):
    if isinstance(value, np.ndarray):
        hash_code = hash(tuple(value.flatten()))
    else:
        hash_code = hash(tuple(value.toarray().flatten()))
    return (a * hash_code + b) % num_buckets




def create_lsh_buckets(num_buckets):
    buckets = {}
    for i in range(num_buckets):
        buckets[i] = []
    return buckets

def lsh_education(letter_range, min_awards, num_hash_functions, num_buckets, similarity_threshold, min_dblp_record, max_dblp_record):

    dataset = pandas.read_csv('BigDataset.csv', encoding='unicode_escape')

    # Filter dataset based on criteria
    filtered = dataset[
        (dataset["Surname"].apply(lambda x: letter_range[0] <= x[0] <= letter_range[1])) &
        (dataset["#Awards"] > min_awards) &
        (min_dblp_record <= dataset["#DBLP_Record"]) & (dataset["#DBLP_Record"] <= max_dblp_record)
        ]

    if filtered.empty:
        print("No matching records found for the given criteria.")
        return []

    # Filtrarei ta rows me keno education
    filtered = filtered[pandas.notna(filtered["Education"])]

    if filtered.empty:
        print("No records with non-blank education values found for the given criteria.")
        return []

    # Compute TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(filtered["Education"])

    # Dimiourgei hash functions kai buckets gia LSH
    hash_functions = create_hash_functions(num_hash_functions, num_buckets)
    buckets = create_lsh_buckets(num_buckets)

    similar_scientists = set()

    for i, scientist in filtered.iterrows():
        education = tfidf_vectorizer.transform([scientist["Education"]])
        hash_values = [hash_value(education, a, b, num_buckets) for a, b in hash_functions]

        for hash_val in hash_values:
            buckets[hash_val].append(i)

    for hash_function in hash_functions:
        h_val = hash_value(education, hash_function[0], hash_function[1], num_buckets)
        similar_scientists.update(buckets[h_val])

    similar_scientists = list(similar_scientists)
    results = []
    similar_scientists_sorted = sorted(similar_scientists)

    for i in range(len(similar_scientists_sorted) - 1):
        scientist_index_1 = similar_scientists_sorted[i]
        scientist_index_2 = similar_scientists_sorted[i + 1]
        education1 = tfidf_vectorizer.transform([filtered.loc[scientist_index_1]["Education"]])
        education2 = tfidf_vectorizer.transform([filtered.loc[scientist_index_2]["Education"]])

        similarity = cosine_similarity(education1, education2)

        if similarity >= similarity_threshold:
            results.append(scientist_index_1)

    if len(similar_scientists_sorted) > 0:
        results.append(similar_scientists_sorted[-1])

    return results





class Node:
    def __init__(self, value, bounds):
        self.value = value
        self.children = []
        self.bounds = bounds


class QuadTree:
    def __init__(self, bounds, capacity=4):
        self.root = None
        self.bounds = bounds
        self.capacity = capacity

    def insert(self, value, point):
        if not self.root:
            self.root = Node(value, self.bounds)
            return

        node = self.root
        while len(node.children) > 0:
            quadrant = self.get_quadrant(point, node.bounds)
            if quadrant >= len(node.children):
                for _ in range(quadrant - len(node.children) + 1):
                    node.children.append(None)
            if node.children[quadrant] is None:
                node.children[quadrant] = Node(value, node.bounds)
                return
            node = node.children[quadrant]

        node.children.append(Node(value, node.bounds))

    def search_node(self, node, surname, awards, dblp_record):
        if not node:
            return None
        if node.value == (surname, awards, dblp_record):
            return node
        quadrant = self.get_quadrant((awards, awards), node.bounds)
        if quadrant < len(node.children):
            return self.search_node(node.children[quadrant], surname, awards, dblp_record)
        return None

    def update_node(self, current_surname, current_awards, current_dblp_record, new_surname, new_awards, new_dblp_record):
        node_to_update = self.search_node(self.root, current_surname, current_awards, current_dblp_record)
        if node_to_update:
            node_to_update.value = (new_surname, new_awards, new_dblp_record)
            return True
        return False

    def delete(self, surname, awards, dblp_record):
        node_to_delete = self.search_node(self.root, surname, awards, dblp_record)
        if node_to_delete:
            node = self.root
            parent = None
            while node and node.value != (surname, awards, dblp_record):
                quadrant = self.get_quadrant((awards, awards), node.bounds)
                parent = node
                node = node.children[quadrant]
            if parent:
                parent.children.remove(node)
                if len(parent.children) == 0:
                    parent.children = []
            else:
                self.root = None
            return True

        return False

    def get_quadrant(self, point, bounds):
        mid_x = bounds[0][0] + (bounds[1][0] - bounds[0][0]) / 2
        mid_y = bounds[0][1] + (bounds[1][1] - bounds[0][1]) / 2

        if point[0] < mid_x:
            if point[1] < mid_y:
                return 0
            return 1
        if point[1] < mid_y:
            return 2
        return 3


    def custom_search(self, letter_range, min_awards, min_dblp_record, max_dblp_record):
        filtered = filter_scientists(letter_range, min_awards, min_dblp_record, max_dblp_record)
        results = []
        for i, scientist in filtered.iterrows():
            node = self.root
            while node is not None:
                if len(node.children) == 0:
                    results.append(scientist)
                    break
                quadrant =self.get_quadrant((scientist["#Awards"], scientist["#Awards"]), node.bounds)
                node = node.children[quadrant]
        return results

    def print_tree(self, node, level=0):
        if node:
            print(" " * level + str(node.value))
            for child in node.children:
                self.print_tree(child, level + 1)




def main():
    bounds = [(0, 0), (98, 98)]
    tree = QuadTree(bounds)

    while True:
        print("Menu:")
        print("0) Exit the program")
        print("1) Make the quad-tree")
        print("2) Print the quad-tree")
        print("3) Insert a node in the quad-tree")
        print("4) Search the quad-tree for a certain node")
        print("5) Select a node to delete")
        print("6) Update a node with new values")
        print("7) Query search the quad-tree")

        choice = input("Enter your choice: ")

        if choice == "0":
            print("Exiting the program.")
            break
        elif choice == "1":
            names = dataset["Surname"].tolist()
            awards = dataset["#Awards"].tolist()
            dblp_records = dataset["#DBLP_Record"].tolist()

            start_time = timer()
            for name, award, dblp_record in zip(names, awards, dblp_records):
                tree.insert((name, award, dblp_record), (award, award, dblp_record))
            end_time = timer()

            execution_time = end_time - start_time
            print("Quad-tree created.")
            print("\nCreation of the tree executed in:", execution_time, "seconds")
        elif choice == "2":
            tree.print_tree(tree.root)
        elif choice == "3":
            surname = input("Enter the surname to insert: ")
            awards = int(input("Enter the number of awards to insert: "))
            dblp_record = input("Enter the DBLP record to insert: ")
            start_time = timer()
            tree.insert((surname, awards, dblp_record), (awards, awards, dblp_record))
            end_time = timer()
            execution_time = end_time - start_time
            print(f"Inserted ({surname}, {awards}, {dblp_record}) into the quad-tree.")
            print("\nNode inserted in: ", execution_time, " seconds")
            tree.print_tree(tree.root)
        elif choice == "4":
            surname = input("Enter the surname to search for: ")
            awards = int(input("Enter the number of awards to search for: "))
            dblp_record = input("Enter the DBLP record to search for: ")
            start_time = timer()
            found_node = tree.search_node(tree.root, surname, awards, dblp_record)
            end_time = timer()
            execution_time = end_time - start_time
            if found_node:
                print("Node found:", found_node.value)
            else:
                print("Node not found")
            print("\nFound node in: ", execution_time, " seconds")

        elif choice == "5":
            surname = input("Enter the surname to delete: ")
            awards = int(input("Enter the number of awards to delete: "))
            dblp_record = input("Enter the DBLP record to delete: ")
            start_time = timer()
            deleted = tree.delete(surname, awards, dblp_record)
            end_time = timer()
            execution_time = end_time - start_time
            if deleted:
                print(f"Deleted ({surname}, {awards}, {dblp_record}) from the quad-tree.")
                tree.print_tree(tree.root)

            else:
                print("Node not found for deletion")
            print("\nNode deleted in: ", execution_time, " seconds")
        elif choice == "6":
            current_surname = input("Enter the current surname of the node to update: ")
            current_awards = int(input("Enter the current number of awards of the node to update: "))
            current_dblp_record = input("Enter the current DBLP record of the node to update: ")
            new_surname = input("Enter the new surname: ")
            new_awards = int(input("Enter the new number of awards: "))
            new_dblp_record = input("Enter the new DBLP record: ")
            start_time = timer()
            updated = tree.update_node(current_surname, current_awards, current_dblp_record, new_surname, new_awards, new_dblp_record)
            end_time = timer()
            execution_time = end_time - start_time
            if updated:
                print(f"Updated ({current_surname}, {current_awards}, {current_dblp_record}) to ({new_surname}, {new_awards}, {new_dblp_record})")
                tree.print_tree(tree.root)
            else:
                print("Node not found for updating")
            print("\nNode updated in: ", execution_time, " seconds")

        elif choice == "7":
            letter_range = input("Enter the letter range (e.g. [A,G]): ")
            letter_range = list(letter_range.strip("[]").split(","))
            letter_range = [letter_range[0], letter_range[-1]]
            min_awards = int(input("Enter the minimum number of awards "))
            min_dblp_record = input("Enter the minimum DBLP record ")
            max_dblp_record = input("Enter the maximum DBLP record ")

            bounds = [(0, 0), (98, 98)]
            tree = QuadTree(bounds)

            start_time = timer()
            results = tree.custom_search(letter_range, min_awards, min_dblp_record, max_dblp_record)
            end_time = timer()

            print("Scientists with surnames in the range {range}, more than {awards} awards, and DBLP record between {min_dblp} and {max_dblp}:".format(
                range=letter_range, awards=min_awards, min_dblp=min_dblp_record, max_dblp=max_dblp_record))
            for scientist in results:
                print(scientist["Surname"], scientist["#Awards"], scientist["#DBLP_Record"])

            num_hash_functions = 50  # Number of hash functions
            num_buckets = 1000  # Number of buckets

            similarity_percentage = float(input("Enter the similarity percentage (0.0-100.0): ")) / 100.0
            similar_education_results = lsh_education(letter_range, min_awards, num_hash_functions, num_buckets, similarity_percentage, min_dblp_record, max_dblp_record)
            print("Scientists with similar education paragraphs:")
            for scientist_index in similar_education_results:
                scientist = dataset.loc[scientist_index]
                print(scientist["Surname"], scientist["Education"])
                print(scientist_index)


            execution_time = end_time - start_time
            print("\nQuery search executed in: ", execution_time, " seconds")

if __name__ == "__main__":
    main()