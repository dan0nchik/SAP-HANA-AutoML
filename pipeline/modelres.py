class ModelBoard:
    def __init__(self, model, train_accuracy: int, preprocessor: dict):
        self.model = model
        self.train_accuracy = train_accuracy
        self.valid_accuracy = 0
        self.preprocessor = preprocessor

    def add_valid_acc(self, accuracy):
        self.valid_accuracy = accuracy
