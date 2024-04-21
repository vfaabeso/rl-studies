from p_queue import PQueue

pq = PQueue()
pq.push_replace(9, 'A')
pq.push_replace(1, 'B')
pq.push_replace(3, 'C')
pq.push_replace(2, 'D')
print(pq)
pq.push_replace(7, 'B')
print(pq)
pq.push_replace(4, 'B')
print(pq)
print(pq.pop_first())
print(pq.pop_last())