class EarlyStopping:
    def __init__(self, tolerance=5, max_delta=0):

        self.tolerance = tolerance
        self.max_delta = max_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_accuracy, test_accuracy):
        # print(f"[DEBUG] Overfitting Check: {train_accuracy}, {test_accuracy}")
        if (train_accuracy - test_accuracy) > self.max_delta:
            self.counter += 1
            print(f"[INFO] Overfitting counter: {self.counter}")
            if self.counter >= self.tolerance:
                self.early_stop = True
