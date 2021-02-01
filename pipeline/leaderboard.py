class Leaderboard:
    def __init__(self):
        self.params_list = []

    def add(self, params):
        self.params_list.append(params)

    def show(self):
        print("=========== LEADERBOARD ===========")
        place = 1
        for param in sorted(self.params_list):
            print(place, ".", param)
