class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        # with torch.cuda.stream(self.stream):
        #     self.next_data = self.next_data.cuda(non_blocking=True)

    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data

"""#Example of how to use this to pre load data
    prefetcher = data_prefetcher(train_loader)
    data = prefetcher.next()
    i = 0
    while data is not None:
        print(i, len(data))
        i += 1
        data = prefetcher.next()
"""