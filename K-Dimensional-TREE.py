import pandas as pd
import numpy as np
from math import floor
from timeit import default_timer as timer
from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHash, MinHashLSH


pd.options.display.max_rows = 9999

# diavazei pdf me panda

df = pd.read_csv('BigDataset.csv')

# dblp to kanei float
df['#DBLP_Record'] = pd.to_numeric(df['#DBLP_Record'], errors='coerce')


points_list_awards = df[[df.columns[0], df.columns[1], df.columns[3]]].values.tolist()
# surname awards dlpd
# surname awards education


class Nodes:
    def __init__(self, x, y, z, left=None, right=None, assoc=None):
        self.x = x
        self.y = y
        self.z = z
        self.xrangemin = None
        self.xrangemax = None
        self.yrmin = None
        self.yrmax = None
        self.zrmin = None
        self.zrmax = None
        self.left = left
        self.right = right
        self.assoc = assoc


def make_kd_tree(points_list, k_dimension, depth):
    if len(points_list) == 0:
        return None

    axis = depth % k_dimension

    if len(points_list) % 2 == 0:
        median = int(len(points_list) / 2)
    else:
        median = floor(len(points_list) / 2)

    if axis == 0:
        sorted_list = sorted(points_list, key=lambda x: x[0])
        root = Nodes(sorted_list[median][0], sorted_list[median][1], sorted_list[median][2])
        s_list = sorted(points_list, key=lambda x: x[1])
        t_list = sorted(points_list, key=lambda x: x[2])
        root.xrangemin = sorted_list[0][0]
        root.xrangemax = sorted_list[-1][0]
        root.yrmin = s_list[0][1]
        root.yrmax = s_list[-1][1]
        root.zrmin = t_list[0][2]
        root.zrmax = t_list[-1][2]
        root.left = make_kd_tree(sorted_list[:median], k_dimension, depth + 1)
        root.right = make_kd_tree(sorted_list[median + 1:], k_dimension, depth + 1)
    elif axis == 1:
        sorted_list = sorted(points_list, key=lambda x: x[1])
        root = Nodes(sorted_list[median][0], sorted_list[median][1], sorted_list[median][2])
        s_list = sorted(points_list, key=lambda x: x[0])
        t_list = sorted(points_list, key=lambda x: x[2])
        root.xrangemin = s_list[0][0]
        root.xrangemax = s_list[-1][0]
        root.yrmin = sorted_list[0][1]
        root.yrmax = sorted_list[-1][1]
        root.zrmin = t_list[0][2]
        root.zrmax = t_list[-1][2]
        root.left = make_kd_tree(sorted_list[:median], k_dimension, depth + 1)
        root.right = make_kd_tree(sorted_list[median + 1:], k_dimension, depth + 1)
    elif axis == 2:
        sorted_list = sorted(points_list, key=lambda x: x[2])
        root = Nodes(sorted_list[median][0], sorted_list[median][1], sorted_list[median][2])
        s_list = sorted(points_list, key=lambda x: x[0])
        t_list = sorted(points_list, key=lambda x: x[1])
        root.xrangemin = s_list[0][0]
        root.xrangemax = s_list[-1][0]
        root.yrmin = t_list[0][1]
        root.yrmax = t_list[-1][1]
        root.zrmin = sorted_list[0][2]
        root.zrmax = sorted_list[-1][2]
        root.left = make_kd_tree(sorted_list[:median], k_dimension, depth + 1)
        root.right = make_kd_tree(sorted_list[median + 1:], k_dimension, depth + 1)

    return root

def printTree(root, string=" "):
    if root is None:
        return None

    if root:
        print(
            string
            + "X:" + str(root.x)
            + " "
            + "Y:" + str(root.y)
            + " "
            + "Z:" + str(root.z)
            + " "
            + "Range X: " + str(root.xrangemin) + " - " + str(root.xrangemax)
            + " "
            + "Range Y: " + str(root.yrmin) + " - " + str(root.yrmax)
            + " "
            + "Range Z: " + str(root.zrmin) + " - " + str(root.zrmax)
        )
        printTree(root.left, string + "-leftchild-")
        printTree(root.right, string + "-rightchild-")


def searchKdTree(root, point, k_dimension, depth):
    if root is None:
        return print("Could not find the node. Please try again.")
    elif (root.x == point.x and root.y == point.y and root.z == point.z):
        return print("Search complete. Node found.")

    axis = depth % k_dimension

    if axis == 0:
        if point.x < root.x:
            searchKdTree(root.left, point, k_dimension, depth + 1)
        else:
            searchKdTree(root.right, point, k_dimension, depth + 1)
    elif axis == 1:
        if point.y < root.y:
            searchKdTree(root.left, point, k_dimension, depth + 1)
        else:
            searchKdTree(root.right, point, k_dimension, depth + 1)
    elif axis == 2:
        if point.z < root.z:
            searchKdTree(root.left, point, k_dimension, depth + 1)
        else:
            searchKdTree(root.right, point, k_dimension, depth + 1)



def insertKdTree(root, point, k_dimension, depth):
    axis = depth % k_dimension

    if (root.left is None or root.right is None):
        if axis == 0:
            if root.left is None and point.x < root.x:
                point.yrmin = point.y
                point.yrmax = point.y
                point.xrangemin = point.x
                point.xrangemax = point.x
                point.zrmin = point.z
                point.zrmax = point.z
                root.left = point
                return print("Node inserted.")
            elif (root.left is not None and point.x < root.x):
                insertKdTree(root.left, point, k_dimension, depth + 1)

            if root.right is None and point.x >= root.x:
                point.yrmin = point.y
                point.yrmax = point.y
                point.xrangemin = point.x
                point.xrangemax = point.x
                point.zrmin = point.z
                point.zrmax = point.z
                root.right = point
                return print("Node inserted.")
            elif (root.right is not None and point.x >= root.x):
                insertKdTree(root.right, point, k_dimension, depth + 1)
        elif axis == 1:
            if root.left is None and point.y < root.y:
                point.yrmin = point.y
                point.yrmax = point.y
                point.xrangemin = point.x
                point.xrangemax = point.x
                point.zrmin = point.z
                point.zrmax = point.z
                root.left = point
                return print("Node inserted.")
            elif (root.left is not None and point.y < root.y):
                insertKdTree(root.left, point, k_dimension, depth + 1)

            if root.right is None and point.y >= root.y:
                point.yrmin = point.y
                point.yrmax = point.y
                point.xrangemin = point.x
                point.xrangemax = point.x
                point.zrmin = point.z
                point.zrmax = point.z
                root.right = point
                return print("Node inserted.")
            elif (root.right is not None and point.y >= root.y):
                insertKdTree(root.right, point, k_dimension, depth + 1)
        elif axis == 2:
            if root.left is None and point.z < root.z:
                point.yrmin = point.y
                point.yrmax = point.y
                point.xrangemin = point.x
                point.xrangemax = point.x
                point.zrmin = point.z
                point.zrmax = point.z
                root.left = point
                return print("Node inserted.")
            elif (root.left is not None and point.z < root.z):
                insertKdTree(root.left, point, k_dimension, depth + 1)

            if root.right is None and point.z >= root.z:
                point.yrmin = point.y
                point.yrmax = point.y
                point.xrangemin = point.x
                point.xrangemax = point.x
                point.zrmin = point.z
                point.zrmax = point.z
                root.right = point
                return print("Node inserted.")
            elif (root.right is not None and point.z >= root.z):
                insertKdTree(root.right, point, k_dimension, depth + 1)
    elif (root.left is not None and root.right is not None):
        if axis == 0:
            if point.x < root.x:
                insertKdTree(root.left, point, k_dimension, depth + 1)
            else:
                insertKdTree(root.right, point, k_dimension, depth + 1)
        elif axis == 1:
            if point.y < root.y:
                insertKdTree(root.left, point, k_dimension, depth + 1)
            else:
                insertKdTree(root.right, point, k_dimension, depth + 1)
        elif axis == 2:
            if point.z < root.z:
                insertKdTree(root.left, point, k_dimension, depth + 1)
            else:
                insertKdTree(root.right, point, k_dimension, depth + 1)






def update_range(root, points_list, k_dimension, depth):
    if root is None:
        return None

    if len(points_list) == 0:
        return None

    if len(points_list) % 2 == 0:  # Ypologismos median
        median = int(len(points_list) / 2)
    else:
        median = floor(len(points_list) / 2)

    axis = depth % k_dimension

    if axis == 0:
        sorted_list = sorted(points_list, key=lambda x: x[0])
        s_list = sorted(points_list, key=lambda x: x[1])
        t_list = sorted(points_list, key=lambda x: x[2])
        root.xrangemin = sorted_list[0][0]
        root.xrangemax = sorted_list[-1][0]
        root.yrmin = s_list[0][1]
        root.yrmax = s_list[-1][1]
        root.zrmin = t_list[0][2]
        root.zrmax = t_list[-1][2]
        update_range(root.right, sorted_list[median + 1:], k_dimension, depth + 1)
        update_range(root.left, sorted_list[:median], k_dimension, depth + 1)
    elif axis == 1:
        sorted_list = sorted(points_list, key=lambda x: x[1])
        s_list = sorted(points_list, key=lambda x: x[0])
        t_list = sorted(points_list, key=lambda x: x[2])
        root.xrangemin = s_list[0][0]
        root.xrangemax = s_list[-1][0]
        root.yrmin = sorted_list[0][1]
        root.yrmax = sorted_list[-1][1]
        root.zrmin = t_list[0][2]
        root.zrmax = t_list[-1][2]
        update_range(root.right, sorted_list[median + 1:], k_dimension, depth + 1)
        update_range(root.left, sorted_list[:median], k_dimension, depth + 1)
    elif axis == 2:
        sorted_list = sorted(points_list, key=lambda x: x[2])
        s_list = sorted(points_list, key=lambda x: x[0])
        t_list = sorted(points_list, key=lambda x: x[1])
        root.xrangemin = s_list[0][0]
        root.xrangemax = s_list[-1][0]
        root.yrmin = t_list[0][1]
        root.yrmax = t_list[-1][1]
        root.zrmin = sorted_list[0][2]
        root.zrmax = sorted_list[-1][2]
        update_range(root.right, sorted_list[median + 1:], k_dimension, depth + 1)
        update_range(root.left, sorted_list[:median], k_dimension, depth + 1)




def delete_kd(root, node, depth=0):
    if root is None:
        return None

    # Calculate current dimension (x=0, y=1, z=2)
    current_dim = depth % 3

    # Compare the node based on the current dimension
    if current_dim == 0:
        if node.x > root.x:
            root.right = delete_kd(root.right, node, depth + 1)
        elif node.x < root.x:
            root.left = delete_kd(root.left, node, depth + 1)
        else:
            # Found the node, perform deletion
            if root.right:
                min_right = find_min(root.right, current_dim, depth + 1)
                root.x, root.y, root.z = min_right.x, min_right.y, min_right.z
                root.right = delete_kd(root.right, min_right, depth + 1)
            else:
                return root.left
    elif current_dim == 1:
        # Compare the node based on the y dimension
        if node.y > root.y:
            root.right = delete_kd(root.right, node, depth + 1)
        elif node.y < root.y:
            root.left = delete_kd(root.left, node, depth + 1)
        else:
            # Found the node, perform deletion
            if root.right:
                min_right = find_min(root.right, current_dim, depth + 1)
                root.x, root.y, root.z = min_right.x, min_right.y, min_right.z
                root.right = delete_kd(root.right, min_right, depth + 1)
            else:
                return root.left
    elif current_dim == 2:
        # Compare the node based on the z dimension
        if node.z > root.z:
            root.right = delete_kd(root.right, node, depth + 1)
        elif node.z < root.z:
            root.left = delete_kd(root.left, node, depth + 1)
        else:
            # Found the node, perform deletion
            if root.right:
                min_right = find_min(root.right, current_dim, depth + 1)
                root.x, root.y, root.z = min_right.x, min_right.y, min_right.z
                root.right = delete_kd(root.right, min_right, depth + 1)
            else:
                return root.left

    return root


def find_min(node, dim, depth):
    next_dim = depth % 3
    if next_dim == dim:
        while node.left:
            node = node.left
    else:
        while node.left or node.right:
            if node.left:
                node = node.left
            else:
                node = node.right
    return node




def update_kd_tree(root, old_point, new_point, k_dimension, depth):
    # Afairesh paliou shmeioy
    if old_point in points_list_awards:
        points_list_awards.remove(old_point)

    # Eisagwgh neoy shmeioy
    points_list_awards.append(new_point)


    root = make_kd_tree(points_list_awards, k_dimension, depth)
    update_range(root, points_list_awards, k_dimension, depth)

    return root

def query_range(root, min_letter, max_letter, min_y, min_z, max_z, results):
    if root is None:
        return

    # print(f"Checking node: {root.x}, {root.y}, {root.z}")

    # Check if the current node matches the query criteria
    if (
            min_letter.lower() <= root.x.lower()[0] <= max_letter.lower()[0]
            and (min_y is None or min_y <= root.y)
            and (min_z is None or min_z <= root.z)
            and (max_z is None or max_z >= root.z)
    ):
        # Fetch education information from the original dataset based on the name
        education_info = df[df['Surname'] == root.x]['Education'].values
        if len(education_info) > 0:
            education_info = education_info[0]
            results.append((root.x, root.y, root.z, education_info))

    # Recursively search in left and right subtrees
    if min_letter.lower() <= root.x.lower()[0]:
        # print(f"Going left from {root.x}")
        query_range(root.left, min_letter, max_letter, min_y, min_z, max_z, results)
    if max_letter.lower() >= root.x.lower()[0]:
        # print(f"Going right from {root.x}")
        query_range(root.right, min_letter, max_letter, min_y, min_z, max_z, results)


    if min_y is None or (min_y is not None and root.y is not None and min_y < root.y):
        if root.assoc is not None:
            query_range(root.assoc, min_letter, max_letter, min_y, min_z, max_z, results)

    if root.assoc is not None:
        query_range(root.assoc, min_letter, max_letter, min_y, min_z, max_z, results)

def extract_education(query_results):
    return [result[3] for result in query_results]

def create_minhash(text):
    minhash = MinHash(num_perm=128)
    for word in text.split():
        minhash.update(word.encode('utf-8'))
    return minhash

def lsh_education_similarity(records, similarity_percentage):
    threshold = similarity_percentage / 100.0

    # Create MinHashes
    minhashes = []
    for record in records:
        text = str(record[3])  # Assuming 'Education' is at index 3
        minhash = MinHash(num_perm=128)
        for word in text.split():
            minhash.update(word.encode('utf-8'))
        minhashes.append(minhash)

    # Create LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    for i, minhash in enumerate(minhashes):
        lsh.insert(str(i), minhash)

    # Query LSH index for similar records
    similar_records = set()
    for i, minhash in enumerate(minhashes):
        similar_items = lsh.query(minhash)
        similar_items = [item for item in similar_items if item != str(i)]
        similar_records.update(similar_items)

    # Return similar records
    return [records[int(item)] for item in similar_records]

############################################################################
#-----------------------------------MAIN-----------------------------------#
num_hash_functions = 50  # Number of hash functions
num_buckets = 1000

con = True
while con:

    print("Give a number for each choice below: ")
    print("0) End the program")
    print("1) Create the K-Dimensional tree")
    print("2) Print the K-Dimensional tree")
    print("3) Insert a node")
    print("4) Search for a node")
    print("5) Delete a node")
    print("6) Update a node")
    print("7) Query search")
    try:
        choice = int(input("Choose one of the above by its number:"))
    except ValueError:
        print("Enter a valid integer.")
        choice = 10

    if choice == 1:
        start_time = timer()
        kd_tree_root = make_kd_tree(points_list_awards, 3, 0)
        end_time = timer()
        execution_time = end_time - start_time
        printTree(kd_tree_root)
        print("\n Make tree executed in: ", execution_time, " seconds")
    elif choice == 2:
        printTree(kd_tree_root)
    elif choice == 3:
        x = input("Insert the x value of node to be inserted:")
        y = int(input("Insert the y value of node to be inserted:"))
        z = float(input("Insert the z value of node to be inserted:"))
        point = Nodes(x, y, z)
        start_time = timer()
        insertKdTree(kd_tree_root, point, 3, 0)
        Dict = {'X': [point.x], 'Y': [point.y], 'Z': [point.z]}
        df_point = pd.DataFrame(Dict)
        points_list_awards += df_point.values.tolist()
        update_range(kd_tree_root, points_list_awards, 3, 0)
        end_time = timer()
        execution_time = end_time - start_time
        printTree(kd_tree_root)
        print("\nNode inserted in: ", execution_time, " seconds")
    elif choice == 4:
        x = input("Insert the x value of node to be searched:")
        y = int(input("Insert the y value of node to be searched:"))
        z = float(input("Insert the z value of node to be searched:"))
        point = Nodes(x, y, z)
        start_time = timer()

        searchKdTree(kd_tree_root, point, 3, 0)
        end_time = timer()
        execution_time = end_time - start_time
        print("\n Node search executed in: ", execution_time, " seconds")

    elif choice == 5:
        x = input("Insert the x value of node to be deleted:")
        y = int(input("Insert the y value of node to be deleted:"))
        z = float(input("Insert the z value of node to be searched:"))
        point = Nodes(x, y, z)

        start_time = timer()
        if [point.x, point.y, point.z] in points_list_awards:
            points_list_awards.remove([point.x, point.y, point.z])


            root = delete_kd(kd_tree_root, point, 0)

            end_time = timer()
            execution_time = end_time - start_time


            update_range(kd_tree_root, points_list_awards, 3, 0)

            kd_tree_root = make_kd_tree(points_list_awards, 3, 0)
            print("Node updated in: ", execution_time, " seconds")
            print("Tree after deleting the node:")
            printTree(kd_tree_root)


    elif choice == 6:
        print("Update a node:")
        old_x = input("Enter old X coordinate: ")
        old_y = int(input("Enter old Y coordinate: "))
        old_z = float(input("Enter old Z coordinate: "))

        new_x = input("Enter new X coordinate: ")
        new_y = int(input("Enter new Y coordinate: "))
        new_z = float(input("Enter new Z coordinate: "))

        old_point = [old_x, old_y, old_z]
        new_point = [new_x, new_y, new_z]

        start_time = timer()
        kd_tree_root = update_kd_tree(kd_tree_root, old_point, new_point, 3, 0)
        end_time = timer()
        execution_time = end_time - start_time
        print("Node updated in: ", execution_time, " seconds")
        printTree(kd_tree_root)





    elif choice == 7:
        min_letter = input("Enter the minimum letter limit from A to Z, a to z:")
        max_letter = input("Enter the maximum letter limit from A to Z, a to z:")
        min_y = int(input("Enter the minimum awards value:"))
        min_z = int(input("Enter the minimum DBLPRecord value:"))
        max_z = int(input("Enter the maximum DBLPRecord value:"))
        similarity_percentage = float(input("Enter the similarity threshold (0 to 100): "))
        start_time = timer()
        query_results = []

        query_range(kd_tree_root, min_letter, max_letter, min_y, min_z, max_z, query_results)

        print(query_results)

        print("\nQuery Results:")
        for result in query_results:
            print(result)

        education_list = extract_education(query_results)
        similar_records = lsh_education_similarity(query_results, similarity_percentage)
        print("Similar Education Records:")
        for record_pair in similar_records:
            print(record_pair)

        end_time = timer()
        execution_time = end_time - start_time
        print("Node found in: ", execution_time, " seconds")

    elif choice == 10:
        con = False
    else:
        con = False
