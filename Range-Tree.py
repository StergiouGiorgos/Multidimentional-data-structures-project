import pandas as pd
from math import floor
from timeit import default_timer as timer
from datasketch import MinHash, MinHashLSH

pd.options.display.max_rows = 9999

#Dhmiourgia dataframe me vasei to csv poy exei Surname, #ofAwards,#DBLP_Record kai education gia tous scientists
df = pd.read_csv('BigDataset.csv', encoding='unicode_escape')

df['#DBLP_Record'] = pd.to_numeric(df['#DBLP_Record'], errors='coerce')

#Antigrafh twn dyo stylwn surname,#ofawards kai #DBLP_Record se mia lista poy tha apotelesei eisodo sth dimiourgia tou range-tree
names_awards_dblp_list = df[[df.columns[0], df.columns[1], df.columns[3]]].values.tolist()

class Node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.left = None
        self.right = None
        self.minr = None
        self.maxr = None
        self.assoc = None

#Synarthsh gia dhmioyrgia 3d-RangeTree
def makeRange3dTree(points_list, xtree=True, ytree=False, ztree=False):
    if len(points_list) == 0:
        return None

    if len(points_list) % 2 == 0:
        median = int(len(points_list) / 2)
    else:
        median = floor(len(points_list) / 2)

    if xtree:
        sorted_list = sorted(points_list, key=lambda x: x[0])
        root = Node(sorted_list[median][0], sorted_list[median][1], sorted_list[median][2])
        root.minr = sorted_list[0][0]
        root.maxr = sorted_list[-1][0]
        root.left = makeRange3dTree(sorted_list[:median], xtree, ytree, ztree)
        root.right = makeRange3dTree(sorted_list[median + 1:], xtree, ytree, ztree)
    elif ytree:
        sorted_list = sorted(points_list, key=lambda x: x[1])
        root = Node(sorted_list[median][0], sorted_list[median][1], sorted_list[median][2])
        root.minr = sorted_list[0][1]
        root.maxr = sorted_list[-1][1]
        root.left = makeRange3dTree(sorted_list[:median], xtree, ytree, ztree)
        root.right = makeRange3dTree(sorted_list[median + 1:], xtree, ytree, ztree)
    elif ztree:
        sorted_list = sorted(points_list, key=lambda x: x[2])
        root = Node(sorted_list[median][0], sorted_list[median][1], sorted_list[median][2])
        root.minr = sorted_list[0][2]
        root.maxr = sorted_list[-1][2]
        root.left = makeRange3dTree(sorted_list[:median], xtree, ytree, ztree)
        root.right = makeRange3dTree(sorted_list[median + 1:], xtree, ytree, ztree)

    if not ytree and not ztree:
        root.assoc = makeRange3dTree(sorted(points_list, key=lambda x: x[1]), xtree=False, ytree=True, ztree=False)

    return root

#Synarthsh ektyposis 3d-RangeTree
def printTree(root, string=" "):
    if root:
        print(
            string + "X:" + str(root.x) + " " +
            "Y:" + str(root.y) + " " +
            "Z:" + str(root.z) + " " +
            "Range: " + str(root.minr) + " - " + str(root.maxr)
        )
        printTree(root.left, string + "-leftchild-")
        printTree(root.right, string + "-rightchild-")
        printTree(root.assoc, string + "-Ytree-")

#Synarthsh gia eisagogh komboy sto dentro
def insert(root, node, xtree=True, ytree=False, ztree=False):
    if root is None:
        return None

    if xtree:
        if node.x > root.x:
            if root.right is None:
                root.right = node
            else:
                insert(root.right, node, xtree, ytree, ztree)
        elif node.x < root.x:
            if root.left is None:
                root.left = node
            else:
                insert(root.left, node, xtree, ytree, ztree)
        else:
            return None
    elif ytree:
        if node.y > root.y:
            if root.right is None:
                root.right = node
            else:
                insert(root.right, node, xtree, ytree, ztree)
        elif node.y < root.y:
            if root.left is None:
                root.left = node
            else:
                insert(root.left, node, xtree, ytree, ztree)
        else:
            return None
    elif ztree:
        if node.z > root.z:
            if root.right is None:
                root.right = node
            else:
                insert(root.right, node, xtree, ytree, ztree)
        elif node.z < root.z:
            if root.left is None:
                root.left = node
            else:
                insert(root.left, node, xtree, ytree, ztree)
        else:
            return None

    if not ytree and not ztree:
        insert(root.assoc, node, xtree=False, ytree=True, ztree=False)

#Synarthsh ypethynh gia thn enhmerosh ton plhroforion eyrous se kathe dentro meta apo eisagogh h diagrafh
def update_range(root, points_list, xtree=True, ytree=False, ztree=False):
    if root is None:
        return None

    if len(points_list) == 0:
        return None

    if len(points_list) % 2 == 0:  # Ypologismos median
        median = int(len(points_list) / 2)
    else:
        median = floor(len(points_list) / 2)

    if xtree:
        sorted_list = sorted(points_list, key=lambda x: x[0])
        root.minr = sorted_list[0][0]
        root.maxr = sorted_list[-1][0]
        update_range(root.right, sorted_list[median + 1:], xtree, ytree, ztree)
        update_range(root.left, sorted_list[:median], xtree, ytree, ztree)
    elif ytree:
        sorted_list = sorted(points_list, key=lambda x: x[1])
        root.minr = sorted_list[0][1]
        root.maxr = sorted_list[-1][1]
        update_range(root.right, sorted_list[median + 1:], xtree, ytree, ztree)
        update_range(root.left, sorted_list[:median], xtree, ytree, ztree)
    elif ztree:
        sorted_list = sorted(points_list, key=lambda x: x[2])
        root.minr = sorted_list[0][2]
        root.maxr = sorted_list[-1][2]
        update_range(root.right, sorted_list[median + 1:], xtree, ytree, ztree)
        update_range(root.left, sorted_list[:median], xtree, ytree, ztree)

    if not ytree and not ztree:
        update_range(root.assoc, sorted(points_list, key=lambda x: str(x[1])), xtree=False, ytree=True, ztree=False)

#Synarthsh gia daigrafh komboy apo to dentro
def delete(root, node, xtree=True, ytree=False, ztree=False):
    if root is None:
        return None

    # Search for the node to delete
    if xtree:
        if node.x > root.x:
            root.right = delete(root.right, node, xtree, ytree, ztree)
        elif node.x < root.x:
            root.left = delete(root.left, node, xtree, ytree, ztree)
        else:
            # Found the node in the x-tree, switch to y-tree and z-tree for deletion
            root.assoc = delete(root.assoc, node, xtree=False, ytree=True, ztree=False)
    elif ytree:
        if node.y > root.y:
            root.right = delete(root.right, node, xtree, ytree, ztree)
        elif node.y < root.y:
            root.left = delete(root.left, node, xtree, ytree, ztree)
        else:
            # Found the node in the y-tree, switch to x-tree and z-tree for deletion
            root.assoc = delete(root.assoc, node, xtree=True, ytree=False, ztree=False)
    elif ztree:
        if node.z > root.z:
            root.right = delete(root.right, node, xtree, ytree, ztree)
        elif node.z < root.z:
            root.left = delete(root.left, node, xtree, ytree, ztree)
        else:
            # Found the node in the z-tree, switch to x-tree and y-tree for deletion
            root.assoc = delete(root.assoc, node, xtree=True, ytree=True, ztree=False)
    else:
        # Node to delete has been found in all three dimensions, delete it
        if root.left is None:
            return root.right
        elif root.right is None:
            return root.left

        # If the node has two children, find the minimum node in the right subtree
        min_right = find_min(root.right)

        # Copy the values from the minimum node to the current node
        root.x = min_right.x
        root.y = min_right.y
        root.z = min_right.z

        # Delete the minimum node from the right subtree
        root.right = delete(root.right, min_right, xtree, ytree, ztree)

    return root

def find_min(node):
    while node.left is not None:
        node = node.left
    return node

#Synarthsh gia anazhthsh kombou sto dentro
def searchNode(root, x_coord, y_coord, z_coord):
    if root is None:
        return False

    if root.x == x_coord and root.y == y_coord and root.z == z_coord:
        return True

    if root.x >= x_coord:  # Traverse left or right based on x-coordinate
        return searchNode(root.left, x_coord, y_coord, z_coord)
    else:
        return searchNode(root.right, x_coord, y_coord, z_coord)

#Synarthsh gia update komboy sto dentro
def updateNode(root, old_x, old_y, old_z, new_x, new_y, new_z, xtree=True, ytree=False, ztree=False):
    if root is None:
        return None

    if xtree:
        if root.x == old_x and root.y == old_y and root.z == old_z:
            root.x = new_x
            root.y = new_y
            root.z = new_z
        elif root.x < old_x:
            updateNode(root.right, old_x, old_y, old_z, new_x, new_y, new_z, xtree, ytree, ztree)
        elif root.x > old_x:
            updateNode(root.left, old_x, old_y, old_z, new_x, new_y, new_z, xtree, ytree, ztree)
    elif ytree:
        if root.x == old_x and root.y == old_y and root.z == old_z:
            root.x = new_x
            root.y = new_y
            root.z = new_z
        elif root.y < old_y:
            updateNode(root.right, old_x, old_y, old_z, new_x, new_y, new_z, xtree, ytree, ztree)
        elif root.y > old_y:
            updateNode(root.left, old_x, old_y, old_z, new_x, new_y, new_z, xtree, ytree, ztree)
    elif ztree:
        if root.x == old_x and root.y == old_y and root.z == old_z:
            root.x = new_x
            root.y = new_y
            root.z = new_z
        elif root.z < old_z:
            updateNode(root.right, old_x, old_y, old_z, new_x, new_y, new_z, xtree, ytree, ztree)
        elif root.z > old_z:
            updateNode(root.left, old_x, old_y, old_z, new_x, new_y, new_z, xtree, ytree, ztree)

    if not ytree and not ztree:
        updateNode(root.assoc, old_x, old_y, old_z, new_x, new_y, new_z, xtree=False, ytree=True, ztree=False)

#Synarthsh gia ta queries
def query_range(root, min_letter, max_letter, min_y, min_z, max_z, results):
    if root is None:
        return


    # Check if the current node matches the query criteria
    if (
            min_letter.lower() <= root.x.lower()[0] <= max_letter.lower()[0]
            and (min_y is None or min_y <= root.y)
            and (min_z is None or min_z <= root.z)
            and (max_z is None or max_z >= root.z)
    ):
        # Fetch education information from the original dataset based on the name
        education_info = df[df['Surname'].str.lower() == root.x.lower()]['Education'].values
        if len(education_info) > 0:
            education_info = education_info[0]
            results.append((root.x, root.y, root.z, education_info))

    # Recursively search in left and right subtrees
    if min_letter.lower() <= root.x.lower()[0]:
        query_range(root.left, min_letter, max_letter, min_y, min_z, max_z, results)
    if max_letter.lower() >= root.x.lower()[0]:
        query_range(root.right, min_letter, max_letter, min_y, min_z, max_z, results)

    # Recursively search in the Y-tree only if Y is within the specified range
    if min_y is None or (min_y is not None and root.y is not None and min_y < root.y):
        if root.assoc is not None:
            query_range(root.assoc, min_letter, max_letter, min_y, min_z, max_z, results)

    if root.assoc is not None:
        query_range(root.assoc, min_letter, max_letter, min_y, min_z, max_z, results)

#Synarthsh gia eyresh education
def extract_education(query_results):
    return [result[3] for result in query_results]

def create_minhash(text):
    minhash = MinHash(num_perm=128)
    for word in text.split():
        minhash.update(word.encode('utf-8'))
    return minhash

#Synarthsh gia eyresh omoiothton me xrhsh LSH
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

################################################################################################
#MAIN

num_hash_functions = 50  # Number of hash functions
num_buckets = 1000  # Number of buckets
root = None
choice=-1


while choice!=0:
    print("Give a number for each choice below: ")
    print("0) End the program")
    print("1) Create the Range tree")
    print("2) Print the Range tree")
    print("3) Insert a node")
    print("4) Delete a node")
    print("5) Search for a node")
    print("6) Update a node")
    print("7) Query search")
    choice=int(input())
    if choice==1:#CREATE THE RANGE TREE
        root = None
        start_time = timer()
        root = makeRange3dTree(names_awards_dblp_list, xtree=True, ytree=True, ztree=True)
        end_time = timer()
        execution_time = end_time - start_time
        print("\nCreate Range Tree executed in: ", execution_time, " seconds")

    elif choice==2:#PRINT THE RANGE TREE
        print("The Range Tree: ")
        printTree(root)
        start_time = timer()
        end_time = timer()
        execution_time = end_time - start_time
        print("\nCreate Range Tree executed in: ", execution_time, " seconds")

    elif choice == 3:  # INSERT A NODE
        print("Give the coordinates for the node to insert:")
        x_coord_to_insert = str(input("Enter X coordinate: "))
        y_coord_to_insert = int(input("Enter Y coordinate: "))
        z_coord_to_insert = int(input("Enter Z coordinate: "))

        start_time = timer()
        insert_node = Node(x_coord_to_insert, y_coord_to_insert, z_coord_to_insert)
        insert(root, insert_node)
        end_time = timer()

        execution_time = end_time - start_time

        # Update DataFrame with the new node
        new_point = {'Surname': [insert_node.x], '#Awards': [insert_node.y], '#DBLP_Record': [insert_node.z]}
        df = pd.concat([df, pd.DataFrame(new_point)])

        # Rebuild the 3D range tree based on the updated list
        names_awards_dblp_list = df[['Surname', '#Awards', '#DBLP_Record']].values.tolist()
        root = makeRange3dTree(names_awards_dblp_list, xtree=True, ytree=True, ztree=True)

        print("Tree after inserting the node:")
        printTree(root)
        print("\nInsert node executed in:", execution_time, "seconds")


    elif choice == 4:  # DELETE A NODE
        print("Give the coordinates for the node to delete:")
        x_coord_to_delete = str(input("Enter X coordinate to delete: "))
        y_coord_to_delete = int(input("Enter Y coordinate to delete: "))
        z_coord_to_delete = int(input("Enter Z coordinate to delete: "))

        delete_node = Node(x_coord_to_delete, y_coord_to_delete, z_coord_to_delete)

        start_time = timer()

        # Check if the node exists in the list before attempting to delete
        if [delete_node.x, delete_node.y, delete_node.z] in names_awards_dblp_list:
            names_awards_dblp_list.remove([delete_node.x, delete_node.y, delete_node.z])

            # Delete the node from the 3D tree
            root = delete(root, delete_node, xtree=True, ytree=True, ztree=True)

            end_time = timer()
            execution_time = end_time - start_time

            # Update the range for the remaining nodes
            update_range(root, names_awards_dblp_list, xtree=True, ytree=True, ztree=True)

            # Rebuild the 3D tree based on the updated list
            root = makeRange3dTree(names_awards_dblp_list, xtree=True, ytree=True, ztree=True)

            print("Tree after deleting the node:")
            printTree(root)
        else:
            print("Node not found in the list.")

        print("\nDelete node executed in:", execution_time, "seconds")
        df = df[df['Surname'] != delete_node.x]
        # Rebuild the 3D range tree based on the updated list
        names_awards_dblp_list = df[['Surname', '#Awards', '#DBLP_Record']].values.tolist()
        root = makeRange3dTree(names_awards_dblp_list, xtree=True, ytree=True, ztree=True)



    elif choice == 5:#SEARCH FOR A SPECIFIC NODE
        x_coord_to_search = str(input("Enter X coordinate to search: "))
        y_coord_to_search = int(input("Enter Y coordinate to search: "))
        z_coord_to_search = int(input("Enter Z coordinate to search: "))

        start_time = timer()
        result = searchNode(root, x_coord_to_search, y_coord_to_search, z_coord_to_search)
        end_time = timer()
        execution_time = end_time - start_time

        if result:
            print("Node exists in the tree.")
        else:
            print("Node does not exist in the tree.")

        print("\nSearch node executed in:", execution_time, "seconds")

    elif choice == 6:  # UPDATE A NODE IN THE TREE
        print("Update a node:")
        old_x = str(input("Enter old X coordinate: "))
        old_y = int(input("Enter old Y coordinate: "))
        old_z = int(input("Enter old Z coordinate: "))

        new_x = str(input("Enter new X coordinate: "))
        new_y = int(input("Enter new Y coordinate: "))
        new_z = int(input("Enter new Z coordinate: "))

        start_time = timer()
        updateNode(root, old_x, old_y, old_z, new_x, new_y, new_z)
        end_time = timer()
        execution_time = end_time - start_time

        # Update DataFrame based on the updated list
        df.loc[(df['Surname'] == old_x) & (df['#Awards'] == old_y) & (df['#DBLP_Record'] == old_z), ['Surname', '#Awards', '#DBLP_Record']] = [new_x, new_y, new_z]


        # Rebuild the 3D range tree based on the updated list
        names_awards_dblp_list = df[['Surname', '#Awards', '#DBLP_Record']].values.tolist()
        root = makeRange3dTree(names_awards_dblp_list, xtree=True, ytree=True, ztree=True)

        print("Tree after updating the node:")
        printTree(root)
        print("\nUpdate node executed in:", execution_time, "seconds")


    elif choice == 7:
        min_letter = input("Enter the minimum letter limit from A to Z, a to z:")
        max_letter = input("Enter the maximum letter limit from A to Z, a to z:")
        min_y = int(input("Enter the minimum awards value:"))
        min_z = int(input("Enter the minimum DBLPRecord value:"))
        max_z = int(input("Enter the maximum DBLPRecord value:"))
        similarity_percentage = float(input("Enter the similarity threshold (0 to 100): "))

        query_results = []
        query_range(root, min_letter, max_letter, min_y, min_z, max_z, query_results)

        education_list = extract_education(query_results)
        similar_records = lsh_education_similarity(query_results, similarity_percentage)
        print("Similar Education Records:")
        for record_pair in similar_records:
            print(record_pair)
        start_time = timer()
        end_time = timer()
        execution_time = end_time - start_time
        print("\nCreate Range Tree executed in: ", execution_time, " seconds")


    else:
        con = False