import tensorflow as tf

graph = tf.get_default_graph()

print(graph.get_operations())

a = tf.constant(10, name='a') #name useful for tensorboard
b = tf.constant(20,name='b')

c = tf.add(a,b, name='sum')
d = tf.multiply(a,b,name='mul')
e = tf.multiply(c,d,name='e')
#for c, it'll just show the name 'sum' because it's just
#a graph. we have not run the graph yet.
operations = graph.get_operations()
print(operations)

print('Creating session...')
sess = tf.Session()
#since we are just using constants hence we don't need
#to global variable initialize
print(sess.run(e))

for op in graph.get_operations():
    print(op.name)
sess.close()
