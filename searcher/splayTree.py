import random, time

Q, R = 50549, 99661

def hash(seq):
    k = 0
    for ele in seq:
        k = (k * Q) % R
        k = (k + ele) % R
    return k

class Node:
    def __init__(self, key, hash_key, value):
        self.child = [None, None]
        self.parent = None
        self.size = 1
        self.key = key
        self.value = value
        self.hash_key = hash_key
        self.previous = None
        self.next = None

class Splay:
    def __init__(self, limit=10000):
        self.root = None
        self.Table = [None] * R
        self.limit = limit

    def breakLink(self, x):
        left = x.child[0]
        if left is not None:
            left.parent = None
        right = x.child[1]
        if right is not None:
            right.parent = None
        parent = x.parent
        if parent is not None:
            parent.child[int(x == parent.child[1])] = None

        x.child = [None, None]
        x.parent = None

    def get_size(self, node):
        if node is None:
            return 0
        return node.size

    def size(self):
        return self.get_size(self.root)

    def rotate(self, x, d):
        y = x.child[1 - d]
        if y is not None:
            x.child[1 - d] = y.child[d]
            if y.child[d] is not None:
                y.child[d].parent = x
            y.parent = x.parent

        if x.parent is None:
            self.root = y
        else:
            x.parent.child[d ^ int(x != x.parent.child[d])] = y
        if y is not None:
            y.child[d] = x
        x.parent = y
        x.size = self.get_size(x.child[0]) + self.get_size(x.child[1]) + 1
        if y is not None:
            y.size = self.get_size(y.child[0]) + self.get_size(y.child[1]) + 1

    def splay(self, x):
        while x.parent is not None:
            if x.parent.parent is None:
                self.rotate(x.parent, int(x == x.parent.child[0]))
            else:
                d1 = x == x.parent.child[0]
                d2 = x.parent == x.parent.parent.child[0]
                if d1 ^ d2:
                    self.rotate(x.parent, int(d1))
                else:
                    self.rotate(x.parent.parent, int(d1))
                self.rotate(x.parent, int(d2))

    def optimum(self, node, d):
        while node.child[d] is not None:
            node = node.child[d]
        self.splay(node)
        return node

    def minimum(self, root):
        return self.optimum(root, 0)

    def maximum(self, root):
        return self.optimum(root, 1)

    def maintain_size(self):
        if self.get_size(self.root) > self.limit:
            x = self.maximum(self.root)
            self.root = x.child[0]
            if self.root is not None:
                self.root.parent = None
            self.breakLink(x)

    def checkHash(self, node):
        current = self.Table[node.hash_key]
        while current is not None:
            if current.value[1] == node.value[1]:
                return current
            current = current.next
        return None

    def merge(self, left, right):
        if left is None:
            return right
        if right is None:
            return left
        leftRoot = self.maximum(left)
        leftRoot.child[1] = right
        return leftRoot

    def pushNode(self, node):
        x = self.root
        y = None
        while x is not None:
            y = x
            x = x.child[int(node.key >= x.key)]
        node.parent = y
        if y is None:
            self.root = node
        else:
            y.child[int(node.key >= y.key)] = node
        self.splay(node)
        self.maintain_size()

    def push(self, key, value):
        node = Node(key, hash(value[1]), value)
        dup = self.checkHash(node)
        if dup is None:
            self.Table[node.hash_key] = node
        elif node.key < dup.key:
            # Hash Table update
            if dup.next is not None:
                dup.next.previous = node
                node.next = dup.next
                dup.next = None

            if dup.previous is not None:
                dup.previous.next = node
                node.previous = dup.previous
                dup.previous = None
            else:
                self.Table[node.hash_key] = node

            # Remove the duplicates from Splay
            self.splay(dup)
            left = dup.child[0]
            right = dup.child[1]
            self.breakLink(dup)
            self.root = self.merge(left, right)

        x = self.root
        y = None
        while x is not None:
            y = x
            x = x.child[int(key >= x.key)]

        node.parent = y
        if y is None:
            self.root = node
        else:
            y.child[int(node.key >= y.key)] = node
        self.splay(node)
        self.maintain_size()

    def isNotEmpty(self):
        return self.get_size(self.root) > 0

    def pop(self):
        x = self.minimum(self.root)
        if x.previous is not None:
            x.previous.next = x.next
        else:
            self.Table[x.hash_key] = x.next
        if x.next is not None:
            x.next.previous = x.previous
        x.next = None
        x.previous = None

        self.root = x.child[1]
        if self.root is not None:
            self.root.parent = None
        self.breakLink(x)
        return x


'''
if __name__ == '__main__':
    N = 10000000
    Tree = Splay()
    index = list(range(N))
    values = list(range(N))
    random.shuffle(values)

    st = time.time()
    for idx, data in zip(index, values):
        #print('Insert', data)
        Tree.push(data, data)
        #Tree.printTree()
        if Tree.get_size(Tree.root) != Tree.limit:
            print(idx + 1, Tree.get_size(Tree.root))
        
        if idx % 10000 == 1:
            data_1 = Tree.pop()
            data_2 = Tree.pop()
            print('Remove', idx+1, data_1.key, data_2.key, Tree.get_size(Tree.root))
    
    ed = time.time()
    print(ed - st)
    #Tree.printTree()
'''
