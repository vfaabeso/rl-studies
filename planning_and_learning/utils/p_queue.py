class PQueue:
    def __init__(self):
        # items are array [Priority, object]
        # arranged from low to high priority
        self._list = []

    def __repr__(self):
        return str(self._list)

    def __len__(self):
        return len(self._list)

    def push(self, priority: int, obj: any) -> None:
        index = 0
        for idx, item in enumerate(self._list):
            if priority > item[0]:
                index += 1;
            else: break;
        self._list.insert(index, (priority, obj))

    # modified push where if the object exists
    # the priority would be modified instead if the priority is higher
    def push_replace(self, priority: int, obj: any) -> None:
        for idx, (P, item) in enumerate(self._list):
            if item == obj:
                if priority > P:
                    self._list.remove(self._list[idx])
                    self.push(priority, obj)
                return
        # do the normal push instead here
        self.push(priority, obj)

    # pop the lowest priority
    def pop_first(self) -> any:
        obj = self._list[0]
        self._list.remove(obj)
        return obj

    # pop the highest priority
    def pop_last(self) -> any:
        obj = self._list[-1]
        self._list.remove(obj)
        return obj