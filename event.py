class Event():


    def __init__(self, prev_sigma, next_sigma):

        self.previous_sigma = prev_sigma
        self.next_sigma = next_sigma

    def __str__(self) -> str:
        
        return str(list(sorted(self.previous_sigma))) + str(list(sorted(self.next_sigma)))
    
    def __eq__(self, other) -> bool:
        
        return (self.previous_sigma == other.previous_sigma) and (self.next_sigma == other.next_sigma)


if __name__ == '__main__':
    e = Event(set(), set({'A'}))
    eee = Event(set(), set({'A'}))
    
    print(e)

    print(e==eee)    