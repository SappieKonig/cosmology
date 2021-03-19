class data_retriever:

    def __init__(self, indices, threads=16, batch_size=256):
        self.indices = indices
        np.random.shuffle(self.indices)
        self.batch_size = batch_size
        self.threads = threads
        self.current_batch = 0
        self.finished = [False for _ in range(threads)]
        self.running = [False for _ in range(threads)]
        self.input = [0 for _ in range(threads)]
        self.output = [0 for _ in range(threads)]
        threading.Thread(target=self.run).start()

    def __iter__(self):
        yield self.get()

    def run(self):
        while 1:
            if self.current_batch > len(self.indices):
                np.random.shuffle(self.indices)
                self.current_batch = 0
            if False in self.running:
                index = self.running.index(False)
                self.running[index] = True
                workhorses = self.indices[self.current_batch:self.current_batch + self.batch_size]
                parts, records = self.get_data(workhorses)

                thread = threading.Thread(target=self.gather, args=(index, parts, records))
                thread.start()
                self.current_batch += self.batch_size


    def get(self):
        while 1:
            if True in self.finished:
                index = self.finished.index(True)
                self.finished[index] = False
                inputs, outputs = self.input[index], self.output[index]
                self.input[index], self.output[index] = 0, 0
                self.running[index] = False
                return inputs, outputs

    def gather(self, index, parts, records):
        inputs, outputs = self.get_piece_in_n_out(parts, records)
        self.input[index] = inputs
        self.output[index] = outputs
        self.finished[index] = True

    def get_data(self, choices):
        parts = indices_grid[options1[indices[choices, 0] - r], options2[indices[choices, 1] - r]].reshape(-1,
                                                                                                           dia,
                                                                                                           dia,
                                                                                                           depth)
        records = pos_kappa_mass_by_index[choices]
        return parts, records

    def get_piece_in_n_out(self, parts, records):

        inputs, outputs = self.process(parts, records)
        outputs = np.array(outputs).reshape((-1, 1))
        return inputs, outputs

    def process(self, parts, records):
        inputs, outputs = [], []
        for part, record in zip(parts, records):
            _in, _out = self.in_n_out_from_record(part, record)
            inputs += [_in]
            outputs += [_out]
        return inputs, outputs

    def in_n_out_from_record(self, shard, record):
        uniques = pos_kappa_mass_by_index[shard[shard!=-1]]
        distance = np.sum((uniques[:, [0, 1]] - record[[0, 1]])**2, axis=-1)
        candidats = np.argsort(distance)[:input_len]
        winners = uniques[candidats]
        winners[:, [0, 1]] -= record[[0, 1]]
        input_part, output_part = winners[:, [0, 1, 2, 4]], record[-2]
        return input_part, output_part