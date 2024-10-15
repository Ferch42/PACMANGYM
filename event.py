class Event():


    def __init__(self, prev_sigma, next_sigma):

        self.previous_sigma = prev_sigma
        self.next_sigma = next_sigma

    def __str__(self) -> str:
        
        return str(self.previous_sigma) + str(self.next_sigma)


if __name__ == '__main__':
    e = Event(set(), set({'A'}))
    print(e)
        